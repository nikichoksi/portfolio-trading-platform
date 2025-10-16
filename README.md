# Portfolio Insight Agent

An AI-powered conversational agent that provides comprehensive portfolio risk-return analysis using LangChain and modern LLMs.

## Features

- **Natural Language Portfolio Analysis**: Describe your portfolio in plain English and get detailed risk metrics
- **Core Financial Metrics**: Volatility, Beta, Sharpe Ratio, Maximum Drawdown, VaR
- **Narrative Explanations**: AI-generated insights about your portfolio's characteristics
- **Strength & Weakness Identification**: Actionable feedback on your allocation
- **Interactive Streamlit UI**: Easy-to-use conversational interface

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
# Run Streamlit UI
streamlit run src/app.py

# Or use the agent programmatically
python src/agents/portfolio_agent.py
```

## Project Structure

```
portfolio-insight-agent/
├── src/
│   ├── agents/          # LangChain agent implementations
│   ├── core/            # Portfolio analysis logic
│   ├── utils/           # Helper functions
│   └── app.py           # Streamlit application
├── tests/               # Test suite
├── data/                # Sample portfolios and cache
├── config/              # Configuration files
└── CLAUDE.md            # Development guide
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
