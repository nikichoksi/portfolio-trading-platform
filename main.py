"""
Scenario Simulator Agent - Portfolio Stress Testing and Risk Analysis API
Consolidated FastAPI backend with all functionality in a single file
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
import numpy as np
from scipy import stats
import uvicorn
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None
    MONTE_CARLO_SIMULATIONS: int = 10000
    CONFIDENCE_LEVELS: list = [0.90, 0.95, 0.99]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# ============================================================================
# DATA MODELS
# ============================================================================

class Position(BaseModel):
    symbol: str = Field(..., description="Stock/Asset symbol")
    quantity: float = Field(..., gt=0, description="Number of shares/units")
    current_price: float = Field(..., gt=0, description="Current price per unit")

class Portfolio(BaseModel):
    positions: List[Position] = Field(..., description="List of portfolio positions")
    total_value: Optional[float] = Field(None, description="Total portfolio value")
    
    def calculate_total_value(self) -> float:
        return sum(pos.quantity * pos.current_price for pos in self.positions)

class ScenarioRequest(BaseModel):
    portfolio: Portfolio
    scenario_type: Literal["market_crash", "sector_decline", "rate_change", "custom"]
    severity: float = Field(default=0.20, ge=0, le=1)
    duration_days: int = Field(default=30, gt=0)
    affected_sectors: Optional[List[str]] = None
    custom_shock: Optional[Dict[str, float]] = None

class HistoricalStressTestRequest(BaseModel):
    portfolio: Portfolio
    historical_event: Literal["2008_crisis", "covid_crash", "dotcom_bubble", "all"]
    include_recovery: bool = Field(default=True)

class MonteCarloRequest(BaseModel):
    portfolio: Portfolio
    time_horizon_days: int = Field(default=252, gt=0)
    num_simulations: int = Field(default=10000, gt=0, le=100000)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    use_historical_volatility: bool = Field(default=True)

class VaRRequest(BaseModel):
    portfolio: Portfolio
    time_horizon_days: int = Field(default=1, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    method: Literal["parametric", "historical", "monte_carlo"] = Field(default="parametric")

class ScenarioResult(BaseModel):
    scenario_type: str
    severity: float
    initial_value: float
    stressed_value: float
    loss_amount: float
    loss_percentage: float
    positions_impact: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HistoricalStressTestResult(BaseModel):
    event: str
    initial_value: float
    peak_loss_value: float
    peak_loss_percentage: float
    recovery_value: Optional[float] = None
    recovery_percentage: Optional[float] = None
    duration_days: int
    positions_impact: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MonteCarloResult(BaseModel):
    num_simulations: int
    time_horizon_days: int
    initial_value: float
    expected_value: float
    expected_return: float
    std_deviation: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    min_value: float
    max_value: float
    percentiles: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class VaRResult(BaseModel):
    var_value: float
    var_percentage: float
    cvar_value: float
    cvar_percentage: float
    confidence_level: float
    time_horizon_days: int
    method: str
    portfolio_value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ============================================================================
# MARKET DATA SERVICE
# ============================================================================

class MarketDataService:
    """Service for fetching real market data from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = timedelta(minutes=5)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def calculate_volatility(self, symbol: str, period: str = "1y") -> float:
        """Calculate historical volatility from Yahoo Finance data"""
        try:
            data = self.get_historical_data(symbol, period=period)
            if data is None or len(data) < 2:
                return 0.20  # Default 20% volatility
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate annualized volatility
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)  # 252 trading days per year
            
            return float(annual_vol)
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.20
    
    def calculate_expected_return(self, symbol: str, period: str = "1y") -> float:
        """Calculate expected return from historical data"""
        try:
            data = self.get_historical_data(symbol, period=period)
            if data is None or len(data) < 2:
                return 0.08  # Default 8% return
            
            # Calculate total return
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            total_return = (end_price - start_price) / start_price
            
            # Annualize
            days = len(data)
            annual_return = (1 + total_return) ** (252 / days) - 1
            
            return float(annual_return)
        except Exception as e:
            logger.error(f"Error calculating return for {symbol}: {str(e)}")
            return 0.08
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "name": info.get('longName', symbol),
                "current_price": info.get('currentPrice') or info.get('regularMarketPrice'),
                "market_cap": info.get('marketCap'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "beta": info.get('beta', 1.0),
                "52_week_high": info.get('fiftyTwoWeekHigh'),
                "52_week_low": info.get('fiftyTwoWeekLow'),
                "dividend_yield": info.get('dividendYield'),
                "pe_ratio": info.get('trailingPE')
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}

# ============================================================================
# BUSINESS LOGIC SERVICES
# ============================================================================

class ScenarioSimulator:
    """Handles scenario simulations and stress testing"""
    
    def __init__(self):
        self.historical_events = {
            "2008_crisis": {
                "name": "2008 Financial Crisis",
                "peak_loss_percentage": -56.78,
                "recovery_percentage": 101.5,
                "duration_days": 1276,
                "recovery_days": 1095
            },
            "covid_crash": {
                "name": "COVID-19 Market Crash",
                "peak_loss_percentage": -33.92,
                "recovery_percentage": 51.3,
                "duration_days": 181,
                "recovery_days": 147
            },
            "dotcom_bubble": {
                "name": "Dot-Com Bubble Burst",
                "peak_loss_percentage": -78.40,
                "recovery_percentage": 362.8,
                "duration_days": 2755,
                "recovery_days": 1825
            }
        }
    
    def run_stress_test(self, request: ScenarioRequest) -> ScenarioResult:
        portfolio = request.portfolio
        initial_value = portfolio.calculate_total_value()
        
        positions_impact = []
        total_stressed_value = 0
        
        for position in portfolio.positions:
            shock_percentage = self._calculate_shock(
                position.symbol,
                request.scenario_type,
                request.severity,
                request.affected_sectors,
                request.custom_shock
            )
            
            new_price = position.current_price * (1 + shock_percentage)
            new_value = position.quantity * new_price
            total_stressed_value += new_value
            
            positions_impact.append({
                "symbol": position.symbol,
                "initial_price": position.current_price,
                "stressed_price": new_price,
                "price_change_percentage": shock_percentage * 100,
                "initial_value": position.quantity * position.current_price,
                "stressed_value": new_value,
                "value_change": new_value - (position.quantity * position.current_price)
            })
        
        loss_amount = initial_value - total_stressed_value
        loss_percentage = (loss_amount / initial_value) * 100 if initial_value > 0 else 0
        
        return ScenarioResult(
            scenario_type=request.scenario_type,
            severity=request.severity,
            initial_value=initial_value,
            stressed_value=total_stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            positions_impact=positions_impact
        )
    
    def _calculate_shock(
        self,
        symbol: str,
        scenario_type: str,
        severity: float,
        affected_sectors: List[str] = None,
        custom_shock: Dict[str, float] = None
    ) -> float:
        if custom_shock and symbol in custom_shock:
            return custom_shock[symbol]
        
        base_shock = {
            "market_crash": -0.30,
            "sector_decline": -0.25,
            "rate_change": -0.15,
            "custom": -0.20
        }.get(scenario_type, -0.20)
        
        return base_shock * severity
    
    def run_historical_stress_test(self, request: HistoricalStressTestRequest) -> HistoricalStressTestResult:
        portfolio = request.portfolio
        initial_value = portfolio.calculate_total_value()
        
        if request.historical_event == "all":
            results = []
            for event_key in self.historical_events.keys():
                req = HistoricalStressTestRequest(
                    portfolio=portfolio,
                    historical_event=event_key,
                    include_recovery=request.include_recovery
                )
                result = self._run_single_historical_event(req)
                results.append(result)
            return min(results, key=lambda x: x.peak_loss_percentage)
        else:
            return self._run_single_historical_event(request)
    
    def _run_single_historical_event(self, request: HistoricalStressTestRequest) -> HistoricalStressTestResult:
        portfolio = request.portfolio
        initial_value = portfolio.calculate_total_value()
        
        event_data = self.historical_events[request.historical_event]
        peak_loss_pct = event_data["peak_loss_percentage"] / 100
        
        peak_loss_value = initial_value * (1 + peak_loss_pct)
        
        recovery_value = None
        recovery_percentage = None
        
        if request.include_recovery:
            recovery_pct = event_data["recovery_percentage"] / 100
            recovery_value = initial_value * (1 + recovery_pct)
            recovery_percentage = ((recovery_value - initial_value) / initial_value) * 100
        
        positions_impact = []
        for position in portfolio.positions:
            new_price = position.current_price * (1 + peak_loss_pct)
            new_value = position.quantity * new_price
            
            positions_impact.append({
                "symbol": position.symbol,
                "initial_price": position.current_price,
                "stressed_price": new_price,
                "price_change_percentage": peak_loss_pct * 100,
                "initial_value": position.quantity * position.current_price,
                "stressed_value": new_value,
                "value_change": new_value - (position.quantity * position.current_price)
            })
        
        timeline = [
            {
                "day": event_data["duration_days"] // 2,
                "value": peak_loss_value,
                "return_percentage": ((peak_loss_value - initial_value) / initial_value) * 100,
                "event": "Peak Loss"
            }
        ]
        
        if recovery_value:
            timeline.append({
                "day": event_data["recovery_days"],
                "value": recovery_value,
                "return_percentage": ((recovery_value - initial_value) / initial_value) * 100,
                "event": "Recovery"
            })
        
        return HistoricalStressTestResult(
            event=event_data["name"],
            initial_value=initial_value,
            peak_loss_value=peak_loss_value,
            peak_loss_percentage=peak_loss_pct * 100,
            recovery_value=recovery_value,
            recovery_percentage=recovery_percentage,
            duration_days=event_data["duration_days"],
            positions_impact=positions_impact,
            timeline=timeline
        )


class MonteCarloSimulator:
    """Handles Monte Carlo simulations"""
    
    def __init__(self):
        self.default_volatility = 0.20
        self.default_return = 0.08
        self.market_data = MarketDataService()
    
    def run_simulation(self, request: MonteCarloRequest) -> MonteCarloResult:
        portfolio = request.portfolio
        initial_value = portfolio.calculate_total_value()
        
        num_simulations = request.num_simulations
        time_horizon = request.time_horizon_days / 252.0
        
        portfolio_volatility = self._calculate_portfolio_volatility(portfolio)
        portfolio_return = self._calculate_portfolio_return(portfolio)
        
        simulated_values = self._run_monte_carlo(
            initial_value=initial_value,
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            time_horizon=time_horizon,
            num_simulations=num_simulations
        )
        
        expected_value = np.mean(simulated_values)
        expected_return = ((expected_value - initial_value) / initial_value) * 100
        std_deviation = np.std(simulated_values)
        
        var_95 = np.percentile(simulated_values, 5)
        var_99 = np.percentile(simulated_values, 1)
        
        cvar_95 = np.mean(simulated_values[simulated_values <= var_95])
        cvar_99 = np.mean(simulated_values[simulated_values <= var_99])
        
        min_value = np.min(simulated_values)
        max_value = np.max(simulated_values)
        
        percentiles = {
            "1": float(np.percentile(simulated_values, 1)),
            "5": float(np.percentile(simulated_values, 5)),
            "10": float(np.percentile(simulated_values, 10)),
            "25": float(np.percentile(simulated_values, 25)),
            "50": float(np.percentile(simulated_values, 50)),
            "75": float(np.percentile(simulated_values, 75)),
            "90": float(np.percentile(simulated_values, 90)),
            "95": float(np.percentile(simulated_values, 95)),
            "99": float(np.percentile(simulated_values, 99))
        }
        
        return MonteCarloResult(
            num_simulations=num_simulations,
            time_horizon_days=request.time_horizon_days,
            initial_value=initial_value,
            expected_value=float(expected_value),
            expected_return=float(expected_return),
            std_deviation=float(std_deviation),
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            min_value=float(min_value),
            max_value=float(max_value),
            percentiles=percentiles
        )
    
    def _run_monte_carlo(
        self,
        initial_value: float,
        portfolio_return: float,
        portfolio_volatility: float,
        time_horizon: float,
        num_simulations: int
    ) -> np.ndarray:
        random_shocks = np.random.normal(0, 1, num_simulations)
        drift = (portfolio_return - 0.5 * portfolio_volatility**2) * time_horizon
        diffusion = portfolio_volatility * np.sqrt(time_horizon) * random_shocks
        final_values = initial_value * np.exp(drift + diffusion)
        return final_values
    
    def _calculate_portfolio_volatility(self, portfolio) -> float:
        """Calculate portfolio volatility using real market data"""
        total_value = portfolio.calculate_total_value()
        weighted_vol = 0
        for position in portfolio.positions:
            weight = (position.quantity * position.current_price) / total_value
            # Get real volatility from Yahoo Finance
            volatility = self.market_data.calculate_volatility(position.symbol)
            weighted_vol += weight * volatility
        return weighted_vol
    
    def _calculate_portfolio_return(self, portfolio) -> float:
        """Calculate expected portfolio return using real market data"""
        total_value = portfolio.calculate_total_value()
        weighted_return = 0
        for position in portfolio.positions:
            weight = (position.quantity * position.current_price) / total_value
            # Get real return from Yahoo Finance
            expected_return = self.market_data.calculate_expected_return(position.symbol)
            weighted_return += weight * expected_return
        return weighted_return


class VaRCalculator:
    """Handles VaR and CVaR calculations"""
    
    def __init__(self):
        self.default_volatility = 0.20
        self.default_return = 0.08
        self.market_data = MarketDataService()
    
    def calculate_var(self, request: VaRRequest) -> VaRResult:
        portfolio = request.portfolio
        portfolio_value = portfolio.calculate_total_value()
        
        if request.method == "parametric":
            return self._calculate_parametric_var(request, portfolio_value)
        elif request.method == "historical":
            return self._calculate_historical_var(request, portfolio_value)
        elif request.method == "monte_carlo":
            return self._calculate_monte_carlo_var(request, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {request.method}")
    
    def _calculate_parametric_var(self, request: VaRRequest, portfolio_value: float) -> VaRResult:
        portfolio_volatility = self._calculate_portfolio_volatility(request.portfolio)
        time_horizon = request.time_horizon_days / 252.0
        daily_volatility = portfolio_volatility * np.sqrt(time_horizon)
        
        z_score = stats.norm.ppf(1 - request.confidence_level)
        var_value = portfolio_value * abs(z_score) * daily_volatility
        
        cvar_value = portfolio_value * daily_volatility * (
            stats.norm.pdf(z_score) / (1 - request.confidence_level)
        )
        
        return VaRResult(
            var_value=var_value,
            var_percentage=(var_value / portfolio_value) * 100,
            cvar_value=cvar_value,
            cvar_percentage=(cvar_value / portfolio_value) * 100,
            confidence_level=request.confidence_level,
            time_horizon_days=request.time_horizon_days,
            method="parametric",
            portfolio_value=portfolio_value
        )
    
    def _calculate_historical_var(self, request: VaRRequest, portfolio_value: float) -> VaRResult:
        num_observations = 252 * 5
        historical_returns = np.random.normal(
            self.default_return / 252,
            self.default_volatility / np.sqrt(252),
            num_observations
        )
        
        portfolio_returns = historical_returns * request.time_horizon_days
        
        var_percentile = (1 - request.confidence_level) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        var_value = abs(var_return * portfolio_value)
        
        cvar_return = np.mean(portfolio_returns[portfolio_returns <= var_return])
        cvar_value = abs(cvar_return * portfolio_value)
        
        return VaRResult(
            var_value=var_value,
            var_percentage=(var_value / portfolio_value) * 100,
            cvar_value=cvar_value,
            cvar_percentage=(cvar_value / portfolio_value) * 100,
            confidence_level=request.confidence_level,
            time_horizon_days=request.time_horizon_days,
            method="historical",
            portfolio_value=portfolio_value
        )
    
    def _calculate_monte_carlo_var(self, request: VaRRequest, portfolio_value: float) -> VaRResult:
        num_simulations = 10000
        time_horizon = request.time_horizon_days / 252.0
        
        portfolio_volatility = self._calculate_portfolio_volatility(request.portfolio)
        portfolio_return = self._calculate_portfolio_return(request.portfolio)
        
        random_shocks = np.random.normal(0, 1, num_simulations)
        returns = (
            portfolio_return * time_horizon +
            portfolio_volatility * np.sqrt(time_horizon) * random_shocks
        )
        
        portfolio_values = portfolio_value * (1 + returns)
        
        var_percentile = (1 - request.confidence_level) * 100
        var_value = portfolio_value - np.percentile(portfolio_values, var_percentile)
        
        var_threshold = np.percentile(portfolio_values, var_percentile)
        cvar_value = portfolio_value - np.mean(
            portfolio_values[portfolio_values <= var_threshold]
        )
        
        return VaRResult(
            var_value=var_value,
            var_percentage=(var_value / portfolio_value) * 100,
            cvar_value=cvar_value,
            cvar_percentage=(cvar_value / portfolio_value) * 100,
            confidence_level=request.confidence_level,
            time_horizon_days=request.time_horizon_days,
            method="monte_carlo",
            portfolio_value=portfolio_value
        )
    
    def _calculate_portfolio_volatility(self, portfolio) -> float:
        """Calculate portfolio volatility using real market data"""
        total_value = portfolio.calculate_total_value()
        weighted_vol = 0
        for position in portfolio.positions:
            weight = (position.quantity * position.current_price) / total_value
            # Get real volatility from Yahoo Finance
            volatility = self.market_data.calculate_volatility(position.symbol)
            weighted_vol += weight * volatility
        return weighted_vol
    
    def _calculate_portfolio_return(self, portfolio) -> float:
        """Calculate expected portfolio return using real market data"""
        total_value = portfolio.calculate_total_value()
        weighted_return = 0
        for position in portfolio.positions:
            weight = (position.quantity * position.current_price) / total_value
            # Get real return from Yahoo Finance
            expected_return = self.market_data.calculate_expected_return(position.symbol)
            weighted_return += weight * expected_return
        return weighted_return

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Scenario Simulator Agent",
    description="Portfolio stress testing and risk analysis API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
market_data_service = MarketDataService()
scenario_simulator = ScenarioSimulator()
monte_carlo_simulator = MonteCarloSimulator()
var_calculator = VaRCalculator()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Scenario Simulator Agent API",
        "version": "1.0.0",
        "endpoints": {
            "scenarios": "/api/v1/scenarios",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/scenarios/stress-test", response_model=ScenarioResult)
async def stress_test_portfolio(request: ScenarioRequest):
    """Perform stress testing on a portfolio under hypothetical scenarios"""
    try:
        result = scenario_simulator.run_stress_test(request)
        return result
    except Exception as e:
        logger.error(f"Error in stress test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scenarios/historical-stress-test", response_model=HistoricalStressTestResult)
async def historical_stress_test(request: HistoricalStressTestRequest):
    """Test portfolio performance under historical market events"""
    try:
        result = scenario_simulator.run_historical_stress_test(request)
        return result
    except Exception as e:
        logger.error(f"Error in historical stress test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scenarios/monte-carlo", response_model=MonteCarloResult)
async def monte_carlo_simulation(request: MonteCarloRequest):
    """Run Monte Carlo simulation for probabilistic portfolio outcomes"""
    try:
        result = monte_carlo_simulator.run_simulation(request)
        return result
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scenarios/var", response_model=VaRResult)
async def calculate_var(request: VaRRequest):
    """Calculate Value at Risk (VaR) and Conditional VaR (Expected Shortfall)"""
    try:
        result = var_calculator.calculate_var(request)
        return result
    except Exception as e:
        logger.error(f"Error in VaR calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/scenarios/scenario-templates")
async def get_scenario_templates():
    """Get available scenario templates"""
    return {
        "market_crash": {
            "description": "Simulates a broad market crash affecting all positions",
            "default_severity": 0.30,
            "typical_duration": 30
        },
        "sector_decline": {
            "description": "Simulates decline in specific sectors",
            "default_severity": 0.25,
            "typical_duration": 60
        },
        "rate_change": {
            "description": "Simulates impact of interest rate changes",
            "default_severity": 0.15,
            "typical_duration": 90
        },
        "custom": {
            "description": "Custom scenario with user-defined shocks",
            "default_severity": 0.20,
            "typical_duration": 30
        }
    }

@app.get("/api/v1/scenarios/historical-events")
async def get_historical_events():
    """Get available historical stress test events"""
    return {
        "2008_crisis": {
            "name": "2008 Financial Crisis",
            "start_date": "2008-09-15",
            "peak_loss_date": "2009-03-09",
            "peak_loss_percentage": -56.78,
            "recovery_date": "2012-03-05",
            "duration_days": 1276
        },
        "covid_crash": {
            "name": "COVID-19 Market Crash",
            "start_date": "2020-02-19",
            "peak_loss_date": "2020-03-23",
            "peak_loss_percentage": -33.92,
            "recovery_date": "2020-08-18",
            "duration_days": 181
        },
        "dotcom_bubble": {
            "name": "Dot-Com Bubble Burst",
            "start_date": "2000-03-24",
            "peak_loss_date": "2002-10-09",
            "peak_loss_percentage": -78.40,
            "recovery_date": "2007-10-09",
            "duration_days": 2755
        }
    }

# ============================================================================
# MARKET DATA ENDPOINTS
# ============================================================================

@app.get("/api/v1/market-data/price/{symbol}")
async def get_stock_price(symbol: str):
    """Get current stock price from Yahoo Finance"""
    try:
        price = market_data_service.get_current_price(symbol)
        if price is None:
            raise HTTPException(status_code=404, detail=f"Could not fetch price for {symbol}")
        return {
            "symbol": symbol.upper(),
            "current_price": price,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-data/info/{symbol}")
async def get_stock_info(symbol: str):
    """Get comprehensive stock information from Yahoo Finance"""
    try:
        info = market_data_service.get_stock_info(symbol)
        if not info:
            raise HTTPException(status_code=404, detail=f"Could not fetch info for {symbol}")
        return info
    except Exception as e:
        logger.error(f"Error fetching info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-data/volatility/{symbol}")
async def get_stock_volatility(
    symbol: str,
    period: str = "1y"
):
    """Get historical volatility for a stock"""
    try:
        volatility = market_data_service.calculate_volatility(symbol, period=period)
        return {
            "symbol": symbol.upper(),
            "volatility": volatility,
            "volatility_percentage": volatility * 100,
            "period": period,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-data/expected-return/{symbol}")
async def get_stock_expected_return(
    symbol: str,
    period: str = "1y"
):
    """Get expected return for a stock based on historical data"""
    try:
        expected_return = market_data_service.calculate_expected_return(symbol, period=period)
        return {
            "symbol": symbol.upper(),
            "expected_return": expected_return,
            "expected_return_percentage": expected_return * 100,
            "period": period,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating return for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-data/portfolio-metrics")
async def get_portfolio_metrics(
    symbols: str = Query(..., description="Comma-separated stock symbols (e.g., AAPL,MSFT,GOOGL)")
):
    """Get portfolio-level metrics for multiple stocks"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        portfolio_metrics = []
        
        for symbol in symbol_list:
            price = market_data_service.get_current_price(symbol)
            volatility = market_data_service.calculate_volatility(symbol)
            expected_return = market_data_service.calculate_expected_return(symbol)
            
            portfolio_metrics.append({
                "symbol": symbol,
                "current_price": price,
                "volatility": volatility,
                "volatility_percentage": volatility * 100,
                "expected_return": expected_return,
                "expected_return_percentage": expected_return * 100
            })
        
        return {
            "portfolio": portfolio_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

