# Developer Notes

Development guide and technical documentation for the Portfolio Insight Agent project.

## Project Overview

**Portfolio Insight Agent** - An AI-powered conversational agent for portfolio risk analysis using LangChain, Anthropic/OpenAI LLMs, and Streamlit. Part of a larger Multi-Agent Risk Analyzer system, this is the first agent implementation focusing on portfolio insights and risk-return analysis.

## Technology Stack

- **Python 3.10+**
- **LangChain**: Agent orchestration and tool-calling framework
- **Anthropic/OpenAI**: LLM providers for conversational intelligence
- **yfinance**: Real-time financial data retrieval
- **pandas/numpy/scipy**: Quantitative analysis
- **Streamlit**: Interactive web UI
- **plotly**: Data visualizations

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env to add ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### Running the Application
```bash
# Run Streamlit UI (main interface)
streamlit run src/app.py

# Run agent programmatically (CLI mode)
python -m src.agents.portfolio_agent

# Run from src directory
cd src && python -m agents.portfolio_agent
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_portfolio_metrics.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/

# Type checking (if using mypy)
mypy src/
```

## Architecture

### High-Level Structure

```
Portfolio Insight Agent
├── Core Analytics Layer (src/core/)
│   └── portfolio_metrics.py - Financial calculations engine
├── Agent Layer (src/agents/)
│   └── portfolio_agent.py - LangChain agent with tools
├── UI Layer (src/)
│   └── app.py - Streamlit conversational interface
└── Utils (src/utils/)
    └── Helper functions
```

### Key Components

#### 1. Portfolio Analytics (`src/core/portfolio_metrics.py`)

**PortfolioAnalyzer** class:
- Core financial metrics calculation
- Market data fetching via yfinance
- Risk metrics: volatility, beta, VaR, CVaR, max drawdown
- Return metrics: annual return, Sharpe ratio
- Correlation analysis for diversification assessment

Key methods:
- `parse_portfolio(str)`: Converts natural language to holdings dict
- `analyze_portfolio(holdings, period)`: Returns PortfolioMetrics dataclass
- `fetch_price_data(tickers, period)`: Gets historical data
- `calculate_*()`: Individual metric calculations

#### 2. LangChain Agent (`src/agents/portfolio_agent.py`)

**PortfolioInsightAgent** class:
- Wraps PortfolioAnalyzer with conversational AI
- Uses tool-calling pattern with LangChain
- Supports both Anthropic and OpenAI models
- Maintains chat history for multi-turn conversations

Tools exposed to LLM:
- `analyze_portfolio`: Parses and analyzes portfolio descriptions
- `get_correlations`: Returns correlation matrix for tickers

Agent design pattern:
1. User query → LLM reasoning
2. LLM decides to call tool(s)
3. Tools execute (fetch data, calculate metrics)
4. LLM interprets results and generates insights
5. Response returned to user

#### 3. Streamlit UI (`src/app.py`)

Interactive web interface with:
- **Chat tab**: Conversational analysis interface
- **Metrics tab**: Visual dashboards (risk-return scatter, allocation pie, gauges)
- **Sidebar**: Examples, quick analysis, help
- Session state management for chat history
- Real-time visualizations with Plotly

### Data Flow

```
User Input (portfolio description)
    ↓
Streamlit UI / CLI
    ↓
PortfolioInsightAgent (LangChain)
    ↓
LLM decides to call tools
    ↓
PortfolioAnalyzer
    ↓
yfinance API (market data)
    ↓
Calculate metrics (pandas/numpy)
    ↓
Return PortfolioMetrics
    ↓
LLM generates narrative insights
    ↓
Display to user (text + visualizations)
```

### Portfolio Metrics Explained

- **Annual Return**: Annualized returns based on historical data
- **Volatility**: Standard deviation of returns (annualized), measures risk
- **Sharpe Ratio**: Risk-adjusted return (return - risk_free_rate) / volatility
  - >2: Excellent, 1-2: Good, 0-1: Fair, <0: Poor
- **Beta**: Sensitivity to market movements (S&P 500 as benchmark)
  - >1.2: Aggressive, 0.8-1.2: Market-like, <0.8: Defensive
- **Max Drawdown**: Largest peak-to-trough decline in portfolio value
- **VaR (Value at Risk)**: Maximum expected loss at 95% confidence level
- **CVaR (Conditional VaR)**: Expected loss when VaR threshold is exceeded

## Development Guidelines

### Adding New Metrics

1. Add calculation method to `PortfolioAnalyzer` class
2. Update `PortfolioMetrics` dataclass with new field
3. Include in `analyze_portfolio()` method
4. Update `_format_metrics()` in portfolio_agent.py
5. Add visualization in app.py if applicable

### Adding New Tools

1. Define tool function in `_create_tools()` method
2. Create Tool instance with name, func, and description
3. Add to tools list
4. Update system prompt if needed to guide LLM usage

### Model Configuration

- Default model: Anthropic's latest Sonnet model
- Alternative: GPT-4 Turbo for OpenAI
- Temperature: 0.1 (low for factual financial analysis)
- Can be configured via environment variables or agent initialization

### API Keys

- Store in `.env` file (never commit)
- Supports both Anthropic and OpenAI
- Agent auto-detects which provider to use based on available keys
- Rate limits: Be mindful of API usage, especially with market data fetching

### Error Handling

- Market data failures: Gracefully handle missing/delisted tickers
- API errors: Catch and return user-friendly messages
- Parsing errors: Provide format examples when portfolio parsing fails
- Agent errors: `handle_parsing_errors=True` in AgentExecutor

## Testing Strategy

- **Unit tests**: Test individual metric calculations
- **Integration tests**: Test agent tool-calling workflow
- **Mock data**: Use fixtures for market data to avoid API calls in tests
- **Edge cases**: Handle empty portfolios, invalid tickers, extreme allocations

## Future Extensions (Multi-Agent System)

This agent is designed to integrate with 5 additional agents:

1. **Risk Profiler Agent**: Risk tolerance assessment
2. **Scenario Simulator Agent**: What-if analysis and stress testing
3. **Rebalancing Strategist Agent**: Optimization recommendations
4. **Comparative Analytics Agent**: Benchmarking and comparison
5. **Temporal Intelligence Agent**: Time-series risk analysis

Architecture for multi-agent integration:
- Shared `PortfolioAnalyzer` core
- Agent orchestration layer (LangGraph or custom)
- Message passing between agents
- Unified Streamlit UI with agent selection

## Dependencies Notes

- **yfinance**: May have rate limits; consider caching
- **LangChain**: Rapidly evolving; pin versions carefully
- **Anthropic/OpenAI**: Ensure API key has sufficient credits
- **Streamlit**: Hot-reloading during development with `streamlit run`

## Common Issues

1. **"No API key found"**: Check .env file exists and has correct key name
2. **yfinance download fails**: Check ticker symbols are valid and markets are open
3. **Agent parsing errors**: Ensure LLM model supports tool calling
4. **Import errors**: Make sure to run from project root or use `-m` flag

## Project Status

- ✅ Core portfolio metrics implementation
- ✅ LangChain agent with tool-calling
- ✅ Streamlit conversational UI
- ✅ Risk metrics (volatility, Sharpe, beta, VaR, drawdown)
- 🚧 Comprehensive test suite
- 🚧 Historical performance visualization
- 🚧 Multi-agent orchestration layer
- 📋 Additional 5 specialized agents (planned)
