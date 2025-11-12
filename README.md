# Portfolio Trading Platform with Multi-Agent Risk Analysis

A professional trading platform (Robinhood/Zerodha style) with integrated AI-powered risk analysis agents built using LangChain and modern LLMs.

## Multi-Agent Architecture

This platform features specialized AI agents for comprehensive portfolio analysis:

### 1. Portfolio Insight Agent ✅
**Conversational portfolio analysis**

Role: Interprets user portfolios and provides comprehensive risk-return analysis
- Parses natural language portfolio descriptions
- Calculates 7 core metrics (volatility, beta, Sharpe ratio, max drawdown, diversification, sector concentration, top holdings)
- Generates narrative explanations of portfolio characteristics
- Identifies strengths and weaknesses in current allocation
- Interactive visualizations (pie charts, risk metrics)

Example Queries:
- "Analyze my portfolio of AAPL, MSFT, and TSLA"
- "What's the overall risk level of my investments?"
- "How diversified is my portfolio?"

### 2. Risk Profiler Agent ✅
**Risk tolerance assessment and matching**

Role: Evaluates if portfolio aligns with investor's risk profile
- Assesses portfolio risk characteristics (volatility, concentration, sector exposure)
- Compares against risk tolerance benchmarks (conservative/moderate/aggressive)
- Calculates fit scores (0-100) with mismatch identification
- Suggests risk-adjusted alternatives and best-matching profile

Example Queries:
- "Is this portfolio suitable for a conservative investor?"
- "Does my allocation match a moderate risk profile?"
- "What risk profile best suits my portfolio?"

### 3. Scenario Simulator Agent ✅
**What-if analysis and stress testing**

Role: Tests portfolio performance under hypothetical market conditions
- Runs scenario simulations (market crashes, sector declines, rate changes)
- Performs historical stress tests (2008 financial crisis, 2020 COVID crash, 2000 dot-com bubble)
- Monte Carlo simulations for probabilistic outcomes
- Calculates position-level impact breakdown

Example Queries:
- "What happens to my portfolio if tech stocks drop 20%?"
- "How would my investments perform in a 2008-style crisis?"
- "Simulate a 10% market correction"

### 4. Rebalancing Strategist Agent ✅
**Portfolio optimization and adjustment recommendations**

Role: Suggests specific actions to improve risk-return profile
- Optimizes allocations using Modern Portfolio Theory (Sharpe, min-variance, risk-parity)
- Generates specific buy/sell rebalancing trades with dollar amounts
- Identifies overweight/underweight positions relative to targets
- Tax-aware rebalancing suggestions

Example Queries:
- "How should I rebalance to reduce risk by 15%?"
- "Optimize my portfolio to maximize Sharpe ratio"
- "What trades do I need to reach my target allocation?"

### 5. Comparative Analytics Agent ✅
**Cross-sectional analysis and benchmarking**

Role: Compares different investment options, sectors, or strategies
- Side-by-side portfolio comparisons across key metrics
- Benchmarks against market indices (S&P 500, NASDAQ, 60/40, Aggressive Growth)
- Analyzes sector-level risk differences
- Peer portfolio analysis with winner identification

Example Queries:
- "Compare the risk of investing in tech vs utilities"
- "How does my portfolio compare to the S&P 500?"
- "Which is less volatile: AAPL or MSFT?"

### 6. Temporal Intelligence Agent ✅
**Time-based risk analysis and horizon planning**

Role: Analyzes how risk changes across different time periods
- Time-horizon suitability analysis (short/medium/long-term investment)
- Rolling risk metrics over time with trend detection
- Risk evolution analysis across multiple periods
- Appropriate recommendations based on investment timeframe

Example Queries:
- "Is this portfolio suitable for a 10-year investment horizon?"
- "How has my portfolio risk changed over the last year?"
- "What's my 5-year vs 1-year volatility?"

## Platform Features

- **Live Market View**: Real-time stock prices organized by sectors
- **Technical Analysis**: Candlestick charts with pattern detection (Head & Shoulders, Double Top/Bottom, etc.)
- **Quick Trade**: Market and limit orders with instant execution
- **Order Management**: Pending orders with automatic execution when conditions are met
- **Portfolio Overview**: Track holdings, P&L, and performance metrics
- **AI Risk Analysis**: Powered by Portfolio Insight Agent with interactive visualizations

## Quick Start

### Prerequisites

- Python 3.10 or higher
- API key from Anthropic (Claude) or OpenAI

### Installation

```bash
# Clone the repository
cd portfolio-insight-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Application

```bash
# Run the Trading Platform (Main Application)
streamlit run src/app_trading.py

# Or run the standalone Portfolio Insight Agent
streamlit run src/app.py

# Or use the agent programmatically
python src/agents/portfolio_agent.py
```

## Project Structure

```
portfolio-insight-agent/
├── src/
│   ├── agents/              # Multi-agent system (LangChain)
│   │   ├── portfolio_agent.py              # Portfolio Insight Agent
│   │   ├── risk_profiler_agent.py          # Risk Profiler Agent
│   │   ├── scenario_simulator_agent.py     # Scenario Simulator Agent
│   │   ├── rebalancing_strategist_agent.py # Rebalancing Agent
│   │   ├── comparative_analytics_agent.py  # Comparative Analytics Agent
│   │   └── temporal_intelligence_agent.py  # Temporal Intelligence Agent
│   ├── database/            # SQLite database models
│   │   └── models.py
│   ├── services/            # Business logic services
│   │   ├── portfolio_service.py
│   │   └── order_execution.py
│   ├── utils/               # Helper functions
│   │   ├── portfolio_analytics.py
│   │   ├── pattern_detection.py
│   │   └── market_data.py
│   ├── app_trading.py       # Main trading platform
│   └── app.py               # Standalone agent UI
├── tests/                   # Test suite
├── data/                    # Database and cache files
├── config/                  # Configuration files
└── dev-notes.md             # Development guide
```

## Example Usage

```python
from src.agents.portfolio_agent import PortfolioInsightAgent

agent = PortfolioInsightAgent()
result = agent.analyze("Analyze my portfolio: 40% AAPL, 30% MSFT, 30% GOOGL")
print(result)
```

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## License

MIT
