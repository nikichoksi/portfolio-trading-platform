-- Snowflake Schema for Portfolio Trading Platform
-- Market Data Warehouse

-- Create database and schema
CREATE DATABASE IF NOT EXISTS PORTFOLIO_DB;
CREATE SCHEMA IF NOT EXISTS PORTFOLIO_DB.MARKET_DATA;

USE SCHEMA PORTFOLIO_DB.MARKET_DATA;

-- ======================================
-- Stock Price Data (Historical)
-- ======================================
CREATE TABLE IF NOT EXISTS stock_prices (
    ticker VARCHAR(10) NOT NULL,
    price_date DATE NOT NULL,
    open_price DECIMAL(18, 4),
    high_price DECIMAL(18, 4),
    low_price DECIMAL(18, 4),
    close_price DECIMAL(18, 4),
    volume BIGINT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (ticker, price_date)
)
COMMENT = 'Historical stock price data from Yahoo Finance';

-- Index for faster date-range queries
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(price_date);
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker ON stock_prices(ticker);

-- ======================================
-- Portfolio Snapshots (Daily)
-- ======================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id NUMBER AUTOINCREMENT,
    snapshot_date TIMESTAMP_NTZ NOT NULL,
    total_market_value DECIMAL(18, 2),
    total_cost_basis DECIMAL(18, 2),
    total_unrealized_pnl DECIMAL(18, 2),
    total_unrealized_pnl_pct DECIMAL(10, 4),
    num_positions INTEGER,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (snapshot_id)
)
COMMENT = 'Daily portfolio value snapshots';

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date);

-- ======================================
-- Portfolio Positions (Per Snapshot)
-- ======================================
CREATE TABLE IF NOT EXISTS portfolio_positions (
    position_id NUMBER AUTOINCREMENT,
    snapshot_id NUMBER NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    quantity DECIMAL(18, 6),
    avg_cost DECIMAL(18, 4),
    current_price DECIMAL(18, 4),
    market_value DECIMAL(18, 2),
    unrealized_pnl DECIMAL(18, 2),
    unrealized_pnl_pct DECIMAL(10, 4),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (position_id),
    FOREIGN KEY (snapshot_id) REFERENCES portfolio_snapshots(snapshot_id)
)
COMMENT = 'Individual stock positions for each portfolio snapshot';

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_snapshot ON portfolio_positions(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_ticker ON portfolio_positions(ticker);

-- ======================================
-- AI Agent Analysis Results
-- ======================================
CREATE TABLE IF NOT EXISTS agent_analysis (
    analysis_id NUMBER AUTOINCREMENT,
    analysis_date TIMESTAMP_NTZ NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    portfolio_tickers VARCHAR(500),
    analysis_type VARCHAR(50),
    metrics VARIANT,
    recommendations VARIANT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (analysis_id)
)
COMMENT = 'AI agent analysis results and recommendations';

CREATE INDEX IF NOT EXISTS idx_agent_analysis_date ON agent_analysis(analysis_date);
CREATE INDEX IF NOT EXISTS idx_agent_analysis_agent ON agent_analysis(agent_name);

-- ======================================
-- Market Indices (S&P 500, NASDAQ, etc.)
-- ======================================
CREATE TABLE IF NOT EXISTS market_indices (
    index_symbol VARCHAR(10) NOT NULL,
    price_date DATE NOT NULL,
    index_value DECIMAL(18, 4),
    volume BIGINT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (index_symbol, price_date)
)
COMMENT = 'Market index data for benchmarking';

-- ======================================
-- Sector Performance Data
-- ======================================
CREATE TABLE IF NOT EXISTS sector_performance (
    sector_name VARCHAR(50) NOT NULL,
    performance_date DATE NOT NULL,
    avg_return DECIMAL(10, 4),
    volatility DECIMAL(10, 4),
    num_stocks INTEGER,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (sector_name, performance_date)
)
COMMENT = 'Aggregated sector performance metrics';

-- ======================================
-- Useful Views
-- ======================================

-- Latest portfolio snapshot
CREATE OR REPLACE VIEW v_latest_portfolio AS
SELECT
    ps.*,
    pp.ticker,
    pp.quantity,
    pp.market_value,
    pp.unrealized_pnl,
    pp.unrealized_pnl_pct
FROM portfolio_snapshots ps
INNER JOIN portfolio_positions pp ON ps.snapshot_id = pp.snapshot_id
WHERE ps.snapshot_date = (SELECT MAX(snapshot_date) FROM portfolio_snapshots);

-- Portfolio performance over time
CREATE OR REPLACE VIEW v_portfolio_performance AS
SELECT
    snapshot_date,
    total_market_value,
    total_unrealized_pnl,
    total_unrealized_pnl_pct,
    LAG(total_market_value) OVER (ORDER BY snapshot_date) AS prev_value,
    (total_market_value - LAG(total_market_value) OVER (ORDER BY snapshot_date)) AS daily_change,
    ((total_market_value - LAG(total_market_value) OVER (ORDER BY snapshot_date)) /
     LAG(total_market_value) OVER (ORDER BY snapshot_date) * 100) AS daily_change_pct
FROM portfolio_snapshots
ORDER BY snapshot_date DESC;

-- Stock price trends (30-day rolling average)
CREATE OR REPLACE VIEW v_stock_price_trends AS
SELECT
    ticker,
    price_date,
    close_price,
    AVG(close_price) OVER (PARTITION BY ticker ORDER BY price_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma_30d,
    AVG(close_price) OVER (PARTITION BY ticker ORDER BY price_date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW) AS ma_90d
FROM stock_prices
ORDER BY ticker, price_date DESC;

-- ======================================
-- Data Retention Policies
-- ======================================

-- Keep 2 years of daily stock prices
-- Keep 1 year of portfolio snapshots
-- Keep 90 days of agent analysis

-- These can be implemented via Snowflake tasks or Airflow DAGs
