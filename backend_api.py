"""
Complete Portfolio Trading Platform Backend API
FastAPI backend with all multi-agent functionality
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
import uvicorn
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all agents
from agents.portfolio_agent import PortfolioInsightAgent
from agents.risk_profiler_agent import RiskProfilerAgent
from agents.scenario_simulator_agent import ScenarioSimulatorAgent
from agents.rebalancing_strategist_agent import RebalancingStrategistAgent
from agents.comparative_analytics_agent import ComparativeAnalyticsAgent
from agents.temporal_intelligence_agent import TemporalIntelligenceAgent
from agents.smart_news_sentiment_agent import NewsSentimentAgent

# Import services
from services.portfolio_service import PortfolioService
from core.portfolio_metrics import PortfolioAnalyzer

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
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# ============================================================================
# DATA MODELS
# ============================================================================

class AgentRequest(BaseModel):
    query: str = Field(..., description="User query or portfolio description")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional conversation history")

class AgentResponse(BaseModel):
    response: str = Field(..., description="Agent's response")
    agent: str = Field(..., description="Agent name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PortfolioAnalysisRequest(BaseModel):
    portfolio: str = Field(..., description="Portfolio description (e.g., '40% AAPL, 30% MSFT, 30% GOOGL')")
    analysis_type: Literal["full", "risk", "diversification", "performance"] = Field(default="full")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Portfolio Trading Platform API",
    description="Complete multi-agent portfolio analysis and trading platform API",
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

# Initialize agents (lazy initialization)
_agents = {}

def get_agent(agent_name: str):
    """Get or create an agent instance"""
    if agent_name not in _agents:
        try:
            if agent_name == "portfolio" or agent_name == "portfolio_insight":
                _agents[agent_name] = PortfolioInsightAgent()
            elif agent_name == "risk_profiler" or agent_name == "risk-profiler":
                _agents[agent_name] = RiskProfilerAgent()
            elif agent_name == "scenario_simulator" or agent_name == "scenario-simulator":
                _agents[agent_name] = ScenarioSimulatorAgent()
            elif agent_name == "rebalancing" or agent_name == "rebalancing_strategist":
                _agents[agent_name] = RebalancingStrategistAgent()
            elif agent_name == "comparative" or agent_name == "comparative_analytics":
                _agents[agent_name] = ComparativeAnalyticsAgent()
            elif agent_name == "temporal" or agent_name == "temporal_intelligence":
                _agents[agent_name] = TemporalIntelligenceAgent()
            elif agent_name == "news_sentiment" or agent_name == "news-sentiment":
                _agents[agent_name] = NewsSentimentAgent()
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
        except Exception as e:
            logger.error(f"Error initializing agent {agent_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")
    return _agents[agent_name]

# Initialize services
portfolio_service = PortfolioService()
portfolio_analyzer = PortfolioAnalyzer()

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Portfolio Trading Platform API",
        "version": "1.0.0",
        "agents": [
            "portfolio_insight",
            "risk_profiler",
            "scenario_simulator",
            "rebalancing_strategist",
            "comparative_analytics",
            "temporal_intelligence",
            "news_sentiment"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents_initialized": list(_agents.keys())
    }

# ============================================================================
# PORTFOLIO INSIGHT AGENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/agents/portfolio/analyze", response_model=AgentResponse)
async def portfolio_analyze(request: AgentRequest):
    """Analyze portfolio using Portfolio Insight Agent"""
    try:
        agent = get_agent("portfolio")
        response = agent.analyze(request.query, request.chat_history)
        return AgentResponse(
            response=response,
            agent="portfolio_insight"
        )
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agents/portfolio/chat")
async def portfolio_chat(request: AgentRequest):
    """Chat with Portfolio Insight Agent"""
    try:
        agent = get_agent("portfolio")
        response, updated_history = agent.chat(request.query, request.chat_history)
        return {
            "response": response,
            "chat_history": updated_history,
            "agent": "portfolio_insight",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in portfolio chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RISK PROFILER AGENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/agents/risk-profiler/analyze", response_model=AgentResponse)
async def risk_profiler_analyze(request: AgentRequest):
    """Analyze risk profile using Risk Profiler Agent"""
    try:
        agent = get_agent("risk-profiler")
        response = agent.analyze(request.query, request.chat_history)
        return AgentResponse(
            response=response,
            agent="risk_profiler"
        )
    except Exception as e:
        logger.error(f"Error in risk profiling: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SCENARIO SIMULATOR AGENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/agents/scenario-simulator/analyze", response_model=AgentResponse)
async def scenario_simulator_analyze(request: AgentRequest):
    """Run scenario simulation using Scenario Simulator Agent"""
    try:
        agent = get_agent("scenario-simulator")
        response = agent.analyze(request.query, request.chat_history)
        return AgentResponse(
            response=response,
            agent="scenario_simulator"
        )
    except Exception as e:
        logger.error(f"Error in scenario simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# REBALANCING STRATEGIST AGENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/agents/rebalancing/analyze", response_model=AgentResponse)
async def rebalancing_analyze(request: AgentRequest):
    """Get rebalancing recommendations using Rebalancing Strategist Agent"""
    try:
        agent = get_agent("rebalancing")
        response = agent.analyze(request.query, request.chat_history)
        return AgentResponse(
            response=response,
            agent="rebalancing_strategist"
        )
    except Exception as e:
        logger.error(f"Error in rebalancing analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# COMPARATIVE ANALYTICS AGENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/agents/comparative/analyze", response_model=AgentResponse)
async def comparative_analyze(request: AgentRequest):
    """Compare portfolios using Comparative Analytics Agent"""
    try:
        agent = get_agent("comparative")
        response = agent.analyze(request.query, request.chat_history)
        return AgentResponse(
            response=response,
            agent="comparative_analytics"
        )
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TEMPORAL INTELLIGENCE AGENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/agents/temporal/analyze", response_model=AgentResponse)
async def temporal_analyze(request: AgentRequest):
    """Analyze temporal patterns using Temporal Intelligence Agent"""
    try:
        agent = get_agent("temporal")
        response = agent.analyze(request.query, request.chat_history)
        return AgentResponse(
            response=response,
            agent="temporal_intelligence"
        )
    except Exception as e:
        logger.error(f"Error in temporal analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PORTFOLIO SERVICE ENDPOINTS
# ============================================================================

@app.get("/api/v1/portfolio/positions")
async def get_positions():
    """Get all portfolio positions"""
    try:
        positions = portfolio_service.get_all_positions()
        return {
            "positions": positions,
            "total_value": sum(pos["quantity"] * pos["current_price"] for pos in positions),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/portfolio/analysis")
async def get_portfolio_analysis(
    portfolio: str = Query(..., description="Portfolio description")
):
    """Get portfolio analysis"""
    try:
        holdings = portfolio_analyzer.parse_portfolio(portfolio)
        if not holdings:
            raise HTTPException(status_code=400, detail="Could not parse portfolio")
        
        metrics = portfolio_analyzer.analyze_portfolio(holdings)
        return {
            "metrics": {
                "annual_return": metrics.annual_return,
                "annual_volatility": metrics.annual_volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "beta": metrics.beta,
                "max_drawdown": metrics.max_drawdown,
                "var_95": metrics.var_95,
                "diversification_score": metrics.diversification_score
            },
            "holdings": holdings,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UNIVERSAL AGENT ENDPOINT
# ============================================================================

@app.post("/api/v1/agents/{agent_name}/analyze")
async def universal_agent_analyze(
    agent_name: str,
    request: AgentRequest
):
    """Universal endpoint for any agent analysis"""
    valid_agents = {
        "portfolio": "portfolio_insight",
        "portfolio-insight": "portfolio_insight",
        "risk-profiler": "risk_profiler",
        "scenario-simulator": "scenario_simulator",
        "rebalancing": "rebalancing_strategist",
        "rebalancing-strategist": "rebalancing_strategist",
        "comparative": "comparative_analytics",
        "comparative-analytics": "comparative_analytics",
        "temporal": "temporal_intelligence",
        "temporal-intelligence": "temporal_intelligence",
        "news-sentiment": "news_sentiment",
        "news_sentiment": "news_sentiment"
    }
    
    if agent_name not in valid_agents:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}. Valid agents: {list(valid_agents.keys())}")
    
    try:
        agent = get_agent(agent_name)
        # News Sentiment Agent uses run() method instead of analyze()
        if agent_name in ["news-sentiment", "news_sentiment"]:
            # Extract ticker from query (e.g., "AAPL" or "Analyze AAPL")
            import re
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', request.query.upper())
            ticker = ticker_match.group(1) if ticker_match else "AAPL"
            result = agent.run(ticker=ticker, max_articles=10, save_to_json=False)
            # Format result as JSON string for response
            import json
            response = json.dumps(result, indent=2, default=str)
        else:
            response = agent.analyze(request.query, request.chat_history)
        return AgentResponse(
            response=response,
            agent=valid_agents[agent_name]
        )
    except Exception as e:
        logger.error(f"Error in {agent_name} analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NEWS SENTIMENT AGENT ENDPOINTS
# ============================================================================

class NewsSentimentRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    max_articles: Optional[int] = Field(default=10, description="Maximum number of articles to fetch")
    save_to_json: Optional[bool] = Field(default=False, description="Whether to save results to JSON file")

@app.post("/api/v1/agents/news-sentiment/analyze")
async def news_sentiment_analyze(request: NewsSentimentRequest):
    """Analyze news sentiment for a stock ticker using News Sentiment Agent"""
    try:
        agent = get_agent("news-sentiment")
        result = agent.run(
            ticker=request.ticker,
            max_articles=request.max_articles,
            save_to_json=request.save_to_json
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in news sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ERROR HANDLING
# ============================================================================

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

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "backend_api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

