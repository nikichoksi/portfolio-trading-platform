# Portfolio Trading Platform with Multi-Agent Risk Analysis

A professional trading platform (Robinhood/Zerodha style) with integrated AI-powered risk analysis agents built using LangChain and modern LLMs.

## Multi-Agent Architecture

This platform features specialized AI agents for comprehensive portfolio analysis:

### 1. Portfolio Insight Agent (Implemented)
**Conversational portfolio analysis**

Role: Interprets user portfolios and provides comprehensive risk-return analysis
- Parses natural language portfolio descriptions
- Calculates core metrics (volatility, beta, Sharpe ratio, max drawdown)
- Generates narrative explanations of portfolio characteristics
- Identifies strengths and weaknesses in current allocation
- Interactive visualizations (pie charts, risk metrics)

Example Queries:
- "Analyze my portfolio of AAPL, MSFT, and TSLA"
- "What's the overall risk level of my investments?"
- "How diversified is my portfolio?"

### 2. Risk Profiler Agent (Coming Soon)
**Risk tolerance assessment and matching**

Role: Evaluates if portfolio aligns with investor's risk profile
- Assesses portfolio risk characteristics (volatility, concentration, sector exposure)
- Compares against risk tolerance benchmarks (conservative/moderate/aggressive)
- Identifies mismatches between risk tolerance and actual portfolio risk
- Suggests risk-adjusted alternatives

Example Queries:
- "Is this portfolio suitable for a conservative investor?"
- "Does my allocation match a moderate risk profile?"
- "Am I taking too much risk for my age?"

### 3. Scenario Simulator Agent (Coming Soon)
**What-if analysis and stress testing**

Role: Tests portfolio performance under hypothetical market conditions
- Runs scenario simulations (market crashes, sector declines, rate changes)
- Performs historical stress tests (2008 crisis, COVID crash, dot-com bubble)
- Monte Carlo simulations for probabilistic outcomes
- Calculates conditional VaR and expected shortfall

Example Queries:
- "What happens to my portfolio if tech stocks drop 20%?"
- "How would my investments perform in a recession?"
- "Simulate a 10% market correction"

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
│   ├── agents/              # LangChain agent implementations
│   │   └── portfolio_agent.py
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
