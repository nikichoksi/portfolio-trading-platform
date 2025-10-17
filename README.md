# Portfolio Trading Platform

Complete portfolio management and risk analysis platform with AI agents and scenario simulation.

## Features

### 🤖 AI Portfolio Agent
- Intelligent portfolio management
- Automated trading strategies
- Pattern detection and analysis

### 📊 Scenario Simulator
- **Real Market Data**: Yahoo Finance integration for live prices, volatility, and returns
- **Stress Testing**: Market crash, sector decline, rate change scenarios
- **Historical Stress Tests**: 2008 crisis, COVID crash, dot-com bubble
- **Monte Carlo Simulations**: Probabilistic portfolio outcomes with real volatility
- **Risk Metrics**: VaR and CVaR calculations using actual market data

### 📈 Trading Dashboard
- Real-time portfolio monitoring
- Order execution
- Portfolio analytics and visualizations

## Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Scenario Simulator**
```bash
python src/scenario_simulator.py
```

3. **Access the API**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

4. **Test the API**
```bash
# Health check
curl http://localhost:8000/health

# Get scenario templates
curl http://localhost:8000/api/v1/scenarios/scenario-templates

# Get historical events
curl http://localhost:8000/api/v1/scenarios/historical-events
```

## API Endpoints

### Stress Test Portfolio
```http
POST /api/v1/scenarios/stress-test
```

**Example Request:**
```json
{
  "portfolio": {
    "positions": [
      {"symbol": "AAPL", "quantity": 100, "current_price": 175.50},
      {"symbol": "MSFT", "quantity": 50, "current_price": 380.25}
    ]
  },
  "scenario_type": "market_crash",
  "severity": 0.30,
  "duration_days": 30
}
```

### Historical Stress Test
```http
POST /api/v1/scenarios/historical-stress-test
```

**Example Request:**
```json
{
  "portfolio": {
    "positions": [...]
  },
  "historical_event": "covid_crash",
  "include_recovery": true
}
```

### Monte Carlo Simulation
```http
POST /api/v1/scenarios/monte-carlo
```

**Example Request:**
```json
{
  "portfolio": {"positions": [...]},
  "time_horizon_days": 252,
  "num_simulations": 10000,
  "confidence_level": 0.95
}
```

### Calculate VaR
```http
POST /api/v1/scenarios/var
```

**Example Request:**
```json
{
  "portfolio": {"positions": [...]},
  "time_horizon_days": 1,
  "confidence_level": 0.95,
  "method": "parametric"
}
```

### Get Templates
```http
GET /api/v1/scenarios/scenario-templates
GET /api/v1/scenarios/historical-events
```

### Market Data Endpoints

**Get Current Stock Price**
```http
GET /api/v1/market-data/price/{symbol}
```
Example: `GET /api/v1/market-data/price/AAPL`

**Get Stock Information**
```http
GET /api/v1/market-data/info/{symbol}
```
Returns: name, market cap, sector, beta, P/E ratio, dividend yield

**Get Stock Volatility**
```http
GET /api/v1/market-data/volatility/{symbol}?period=1y
```
Period options: `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`

**Get Expected Return**
```http
GET /api/v1/market-data/expected-return/{symbol}?period=1y
```

**Get Portfolio Metrics**
```http
GET /api/v1/market-data/portfolio-metrics?symbols=AAPL,MSFT,GOOGL
```
Returns metrics for multiple stocks at once

## File Structure

```
portfolio-trading-platform/
├── src/
│   ├── scenario_simulator.py    # Scenario Simulator Agent (FastAPI)
│   ├── portfolio_agent.py       # AI Portfolio Agent
│   ├── app.py                   # Main application
│   ├── app_dashboard.py         # Dashboard app
│   ├── app_trading.py           # Trading app
│   ├── core/
│   │   └── portfolio_metrics.py
│   ├── services/
│   │   ├── portfolio_service.py
│   │   └── order_execution.py
│   ├── utils/
│   │   ├── market_data.py
│   │   ├── portfolio_analytics.py
│   │   ├── pattern_detection.py
│   │   ├── sector_analysis.py
│   │   └── visualizations.py
│   └── database/
│       └── models.py
├── tests/
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## Configuration

Create a `.env` file with:
```env
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Optional API keys
# ALPHA_VANTAGE_API_KEY=your_key
# POLYGON_API_KEY=your_key
# FRED_API_KEY=your_key
```

## Testing

Use the interactive documentation at http://localhost:8000/docs or curl:

```bash
curl -X POST "http://localhost:8000/api/v1/scenarios/var" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "positions": [
        {"symbol": "AAPL", "quantity": 100, "current_price": 175.50}
      ]
    },
    "confidence_level": 0.95,
    "time_horizon_days": 1
  }'
```

## Notes

- **Real market data** from Yahoo Finance (no API key required)
- Automatically fetches current prices, volatility, and returns
- Falls back to defaults if data unavailable or rate-limited
- Monte Carlo uses Geometric Brownian Motion with real volatility
- VaR supports 3 methods: parametric, historical, monte_carlo
- All tested and verified - 100% functional ✅

## Yahoo Finance Integration

The backend now fetches real market data:
- **Current Prices**: Live stock prices
- **Historical Volatility**: Calculated from 1-year price data
- **Expected Returns**: Based on historical performance
- **Stock Info**: Company details, sector, beta, P/E ratio

**Example Usage:**
```bash
# Get AAPL current price
curl http://localhost:8000/api/v1/market-data/price/AAPL

# Get AAPL volatility
curl http://localhost:8000/api/v1/market-data/volatility/AAPL

# Get multiple stocks metrics
curl "http://localhost:8000/api/v1/market-data/portfolio-metrics?symbols=AAPL,MSFT,TSLA"
```

