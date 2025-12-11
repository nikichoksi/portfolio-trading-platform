from langgraph.graph import StateGraph, END
from src.services.snowflake_data_service import SnowflakeDataService
from src.utils.llm_config import get_llm
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Literal
import logging
import os

logger = logging.getLogger(__name__)

class PortfolioInsightAgent:
    """Portfolio Insight Agent using Snowflake as data source"""
    
    def __init__(self, use_snowflake=True, llm_provider: Literal["groq", "anthropic", "openai"] = "groq"):
        """Initialize the PortfolioInsightAgent.
        
        Args:
            use_snowflake: Whether to use Snowflake for data retrieval
            llm_provider: The LLM provider to use ('groq', 'anthropic', or 'openai')
        """
        try:
            self.llm = get_llm(provider=llm_provider, temperature=0)
            logger.info(f"✅ Initialized LLM with provider: {llm_provider}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise
            
        self.use_snowflake = use_snowflake
        
        if use_snowflake:
            try:
                self.data_service = SnowflakeDataService()
                logger.info("✅ Connected to Snowflake data layer")
            except Exception as e:
                logger.warning(f"⚠️ Snowflake not available: {e}")
                self.use_snowflake = False
    
    def analyze(self, query: str) -> Dict:
        """Analyze portfolio using Snowflake data"""
        
        # Parse portfolio from query
        tickers, weights = self._parse_portfolio(query)
        
        if not tickers:
            return {"error": "No valid tickers found"}
        
        # Fetch data from Snowflake
        if self.use_snowflake:
            portfolio_data = self.data_service.get_portfolio_data(tickers)
            metrics = self._calculate_metrics_from_snowflake(
                tickers, 
                weights, 
                portfolio_data
            )
        else:
            # Fallback to yfinance
            metrics = self._calculate_metrics_yfinance(tickers, weights)
        
        # Generate narrative using LLM
        narrative = self._generate_narrative(metrics)
        
        return {
            "tickers": tickers,
            "weights": weights,
            "metrics": metrics,
            "narrative": narrative,
            "data_source": "snowflake" if self.use_snowflake else "yfinance"
        }
    
    def _calculate_metrics_from_snowflake(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        portfolio_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Calculate portfolio metrics using Snowflake data"""
        
        prices_df = portfolio_data['prices']
        sp500_df = portfolio_data['sp500_data']
        fundamentals_df = portfolio_data['fundamentals']
        sector_allocation = portfolio_data['sector_allocation']
        
        # Pivot prices for calculation
        prices_pivot = prices_df.pivot(index='date', columns='ticker', values='close')
        returns = prices_pivot.pct_change().dropna()
        
        # Portfolio returns
        weights_array = np.array([weights.get(t, 0) for t in prices_pivot.columns])
        portfolio_returns = (returns * weights_array).sum(axis=1)
        
        # 1. Volatility
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 2. Returns
        cumulative_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + cumulative_return) ** (252 / len(portfolio_returns)) - 1
        
        # 3. Sharpe Ratio
        risk_free_rate = 0.04
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # 4. Beta (using Snowflake S&P 500 data)
        if not sp500_df.empty:
            sp500_returns = sp500_df.set_index('date')['daily_return'] / 100
            common_dates = portfolio_returns.index.intersection(sp500_returns.index)
            
            if len(common_dates) > 0:
                portfolio_aligned = portfolio_returns.loc[common_dates]
                sp500_aligned = sp500_returns.loc[common_dates]
                
                covariance = np.cov(portfolio_aligned, sp500_aligned)[0][1]
                market_variance = np.var(sp500_aligned)
                beta = covariance / market_variance if market_variance != 0 else 1.0
            else:
                beta = None
        else:
            beta = None
        
        # 5. Maximum Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 6. Diversification Score
        correlation_matrix = returns.corr()
        avg_correlation = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ].mean()
        diversification_score = 1 - avg_correlation
        
        # 7. Sector Concentration
        sector_risk = "HIGH" if max(sector_allocation.values()) > 0.5 else "MODERATE" if max(sector_allocation.values()) > 0.3 else "LOW"
        
        return {
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "beta": float(beta) if beta else None,
            "max_drawdown": float(max_drawdown),
            "diversification_score": float(diversification_score),
            "avg_correlation": float(avg_correlation),
            "sector_allocation": sector_allocation,
            "sector_concentration_risk": sector_risk,
            "num_holdings": len(tickers),
            "data_points": len(portfolio_returns)
        }
    
    def _calculate_metrics_yfinance(
        self, 
        tickers: List[str], 
        weights: Dict[str, float]
    ) -> Dict:
        """Calculate portfolio metrics using yfinance as fallback"""
        import yfinance as yf
        
        try:
            # Fetch historical data
            data = yf.download(tickers, period="1y", progress=False)
            
            if data.empty:
                return self._empty_metrics(tickers, weights)
            
            # Get close prices
            if len(tickers) == 1:
                prices = data['Close'].to_frame()
                prices.columns = tickers
            else:
                prices = data['Close']
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            if returns.empty:
                return self._empty_metrics(tickers, weights)
            
            # Portfolio returns
            weights_array = np.array([weights.get(t, 0) for t in prices.columns])
            portfolio_returns = (returns * weights_array).sum(axis=1)
            
            # 1. Volatility
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # 2. Returns
            cumulative_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + cumulative_return) ** (252 / len(portfolio_returns)) - 1
            
            # 3. Sharpe Ratio
            risk_free_rate = 0.04
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
            
            # 4. Beta (vs S&P 500)
            try:
                sp500 = yf.download("^GSPC", period="1y", progress=False)['Close']
                sp500_returns = sp500.pct_change().dropna()
                common_dates = portfolio_returns.index.intersection(sp500_returns.index)
                
                if len(common_dates) > 20:
                    port_aligned = portfolio_returns.loc[common_dates]
                    sp500_aligned = sp500_returns.loc[common_dates]
                    covariance = np.cov(port_aligned, sp500_aligned)[0][1]
                    market_variance = np.var(sp500_aligned)
                    beta = covariance / market_variance if market_variance != 0 else 1.0
                else:
                    beta = 1.0
            except Exception:
                beta = 1.0
            
            # 5. Maximum Drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 6. Diversification Score
            if len(tickers) > 1:
                correlation_matrix = returns.corr()
                avg_correlation = correlation_matrix.values[
                    np.triu_indices_from(correlation_matrix.values, k=1)
                ].mean()
                diversification_score = 1 - avg_correlation
            else:
                avg_correlation = 1.0
                diversification_score = 0.0
            
            # 7. Sector allocation (simplified)
            sector_allocation = {}
            for ticker in tickers:
                try:
                    info = yf.Ticker(ticker).info
                    sector = info.get('sector', 'Other')
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + weights.get(ticker, 0)
                except Exception:
                    sector_allocation['Other'] = sector_allocation.get('Other', 0) + weights.get(ticker, 0)
            
            sector_risk = "HIGH" if max(sector_allocation.values()) > 0.5 else "MODERATE" if max(sector_allocation.values()) > 0.3 else "LOW"
            
            return {
                "annual_return": float(annual_return),
                "annual_volatility": float(annual_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "beta": float(beta),
                "max_drawdown": float(max_drawdown),
                "diversification_score": float(diversification_score),
                "avg_correlation": float(avg_correlation),
                "sector_allocation": sector_allocation,
                "sector_concentration_risk": sector_risk,
                "num_holdings": len(tickers),
                "data_points": len(portfolio_returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics with yfinance: {e}")
            return self._empty_metrics(tickers, weights)
    
    def _empty_metrics(self, tickers: List[str], weights: Dict[str, float]) -> Dict:
        """Return empty/default metrics when calculation fails"""
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "beta": 1.0,
            "max_drawdown": 0.0,
            "diversification_score": 0.0,
            "avg_correlation": 0.0,
            "sector_allocation": {"Unknown": 1.0},
            "sector_concentration_risk": "UNKNOWN",
            "num_holdings": len(tickers),
            "data_points": 0
        }

    def _parse_portfolio(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """Parse portfolio from natural language"""
        import re
        
        # Extract tickers
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, query)
        
        # Remove common words
        exclude = {'I', 'A', 'THE', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'AND', 'OR', 'MY'}
        tickers = [t for t in tickers if t not in exclude]
        
        # Remove duplicates
        seen = set()
        tickers = [t for t in tickers if not (t in seen or seen.add(t))]
        
        # Extract weights
        weights = {}
        percentage_pattern = r'([A-Z]{1,5})\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(percentage_pattern, query)
        
        if matches:
            for ticker, weight in matches:
                if ticker in tickers:
                    weights[ticker] = float(weight) / 100
        
        # Equal weights if not specified
        if not weights and tickers:
            equal_weight = 1.0 / len(tickers)
            weights = {t: equal_weight for t in tickers}
        
        return tickers, weights
    
    def _generate_narrative(self, metrics: Dict) -> str:
        """Generate analysis narrative using LLM"""
        
        prompt = f"""
You are a financial analyst. Analyze this portfolio and provide insights:

Portfolio Metrics:
- Annual Return: {metrics['annual_return']*100:.2f}%
- Volatility: {metrics['annual_volatility']*100:.2f}%
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Beta: {metrics.get('beta', 'N/A')}
- Max Drawdown: {metrics['max_drawdown']*100:.2f}%
- Diversification Score: {metrics['diversification_score']:.2f}
- Sector Allocation: {metrics['sector_allocation']}
- Sector Concentration Risk: {metrics['sector_concentration_risk']}

Provide:
1. Overall risk assessment (LOW/MODERATE/HIGH)
2. Key strengths
3. Key concerns
4. Recommendations

Keep it concise and actionable.
"""
        
        response = self.llm.invoke(prompt)
        return response.content