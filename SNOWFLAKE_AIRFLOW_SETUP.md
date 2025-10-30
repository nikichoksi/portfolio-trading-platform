# Snowflake & Airflow Integration Guide

This document explains how Snowflake and Apache Airflow are integrated into the portfolio trading platform.

## Architecture Overview

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Streamlit UI   │◄────►│   SQLite     │      │   Snowflake     │
│ (Real-time App) │      │ (Portfolio)  │      │ (Data Warehouse)│
└─────────────────┘      └──────────────┘      └─────────────────┘
                                ▲                        ▲
                                │                        │
                                └────────┬───────────────┘
                                         │
                                  ┌──────────────┐
                                  │   Airflow    │
                                  │ (Orchestration)│
                                  └──────────────┘
                                         │
                                  ┌──────────────┐
                                  │ Yahoo Finance │
                                  └──────────────┘
```

## Components

### 1. Snowflake Data Warehouse

**Purpose**: Store historical market data and analytics results for long-term analysis

**Tables Created**:
- `stock_prices` - Daily OHLCV data for all portfolio tickers
- `portfolio_snapshots` - Daily portfolio valuations
- `portfolio_positions` - Individual position details per snapshot
- `agent_analysis` - AI agent analysis results
- `market_indices` - S&P 500, NASDAQ, etc.
- `sector_performance` - Sector-level aggregations

**Location**: `src/data_warehouse/`

### 2. Apache Airflow

**Purpose**: Orchestrate data pipelines - fetch, transform, load market data

**DAGs Created**:
- `market_data_ingestion` - Fetches stock prices hourly (9 AM - 4 PM ET)

**Location**: `airflow/dags/`

## Setup Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `snowflake-connector-python` - Python connector for Snowflake
- `snowflake-sqlalchemy` - SQLAlchemy support
- `apache-airflow` - Workflow orchestration
- `apache-airflow-providers-snowflake` - Snowflake integration for Airflow

### Step 2: Configure Environment Variables

All credentials are stored securely in `.env`:

```bash
# Snowflake Configuration (already set)
SNOWFLAKE_ACCOUNT=SFEDU02-VOB68402
SNOWFLAKE_USER=BOA
SNOWFLAKE_PASSWORD=Macbook@9619601191
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=PORTFOLIO_DB
SNOWFLAKE_SCHEMA=MARKET_DATA
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

### Step 3: Initialize Snowflake Schema

Run this Python script to create all tables:

```bash
python -c "
from src.data_warehouse.snowflake_connector import SnowflakeWarehouse
from dotenv import load_dotenv

load_dotenv()
warehouse = SnowflakeWarehouse()
warehouse.setup_schema()
print('✓ Snowflake schema created successfully')
"
```

### Step 4: Test Snowflake Connection

```bash
python -c "
from src.data_warehouse.snowflake_connector import SnowflakeWarehouse
from dotenv import load_dotenv

load_dotenv()
warehouse = SnowflakeWarehouse()
conn = warehouse._get_connection()
print(f'✓ Connected to Snowflake: {conn.account}')
warehouse.close()
"
```

### Step 5: Initialize Airflow (Optional)

```bash
# Initialize Airflow database
export AIRFLOW_HOME=~/portfolio-insight-agent/airflow
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start Airflow webserver
airflow webserver --port 8080

# In another terminal, start scheduler
airflow scheduler
```

Access Airflow UI at: http://localhost:8080

## Usage Examples

### Example 1: Insert Stock Prices into Snowflake

```python
from src.data_warehouse.snowflake_connector import SnowflakeWarehouse
import yfinance as yf

# Fetch price data
stock = yf.Ticker("AAPL")
price_data = stock.history(period="1mo")

# Connect to Snowflake
warehouse = SnowflakeWarehouse()
warehouse.setup_schema()  # Ensure tables exist

# Insert data
warehouse.insert_stock_prices("AAPL", price_data)
print("✓ Inserted AAPL price data")

warehouse.close()
```

### Example 2: Query Historical Prices from Snowflake

```python
from src.data_warehouse.snowflake_connector import SnowflakeWarehouse

warehouse = SnowflakeWarehouse()

# Get AAPL prices for last 30 days
prices = warehouse.get_stock_prices(
    ticker="AAPL",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(prices.head())
warehouse.close()
```

### Example 3: Store Portfolio Snapshot

```python
from src.data_warehouse.snowflake_connector import SnowflakeWarehouse
from src.services.portfolio_service import PortfolioService

# Get current portfolio
service = PortfolioService()
summary = service.get_portfolio_summary()
positions = summary['positions']

# Store in Snowflake
warehouse = SnowflakeWarehouse()
warehouse.insert_portfolio_snapshot(summary, positions)
print("✓ Portfolio snapshot saved")

warehouse.close()
```

### Example 4: Get Portfolio Performance History

```python
from src.data_warehouse.snowflake_connector import SnowflakeWarehouse

warehouse = SnowflakeWarehouse()

# Get last 30 days of portfolio history
history = warehouse.get_portfolio_history(days=30)
print(history)

warehouse.close()
```

## Airflow DAG: Market Data Ingestion

**Schedule**: Hourly from 9 AM - 4 PM ET, Monday-Friday

**Tasks**:
1. `fetch_portfolio_tickers` - Get all tickers from SQLite
2. `fetch_market_indices` - Define market indices to track
3. `fetch_stock_prices` - Download price data from Yahoo Finance
4. `load_to_snowflake` - Insert data into Snowflake
5. `validate_data_quality` - Check for stale/anomalous data

**Manual Trigger**:
```bash
airflow dags trigger market_data_ingestion
```

## Benefits of This Architecture

1. **Snowflake**:
   - Stores years of historical data efficiently
   - Fast analytical queries (vs repeated yfinance API calls)
   - Time-travel for portfolio backtesting
   - Scalable to millions of rows

2. **Airflow**:
   - Automated data pipelines
   - Scheduled execution (hourly during market hours)
   - Error handling and retries
   - Data quality validation
   - Email/Slack alerts on failures

3. **Hybrid Approach**:
   - SQLite for real-time transactions (low latency)
   - Snowflake for analytics (historical queries)
   - Best of both worlds

## Next Steps

1. **Enable Airflow** - Run the DAG to start populating Snowflake
2. **Integrate with Agents** - Modify AI agents to query Snowflake for historical data
3. **Add More DAGs**:
   - Portfolio analytics pipeline (daily)
   - Agent training pipeline (weekly)
   - Data quality monitoring (every 6 hours)
4. **Add Visualizations** - Create Streamlit dashboard showing Snowflake data

## Troubleshooting

### Connection Error
```
snowflake.connector.errors.DatabaseError: 250001: Could not connect to Snowflake backend
```

**Solution**: Check your account identifier format. It should be `ORGNAME-ACCOUNTNAME` (e.g., `SFEDU02-VOB68402`)

### Authentication Error
```
snowflake.connector.errors.DatabaseError: 250001: Incorrect username or password
```

**Solution**: Verify credentials in `.env` file

### Schema Not Found
```
Object does not exist, or operation cannot be performed
```

**Solution**: Run `warehouse.setup_schema()` to create database/schema/tables

## Security Notes

- **Never commit `.env`** to git - it contains passwords
- Snowflake credentials are stored in environment variables only
- Config file (`config/snowflake_config.yaml`) uses `${ENV_VAR}` placeholders
- Add `.env` to `.gitignore`

## Support

For issues with:
- Snowflake connection: Check account identifier and credentials
- Airflow DAG: Check logs in Airflow UI
- Data quality: Review validation task output
