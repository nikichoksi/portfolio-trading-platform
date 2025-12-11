-- ============================================
-- SNOWFLAKE INPUT DATA LAYER SCHEMA
-- ============================================

USE DATABASE PORTFOLIO_TRADING;
USE SCHEMA ANALYTICS;

-- ============================================
-- 1. STOCK PRICES (Historical OHLCV Data)
-- ============================================
CREATE OR REPLACE TABLE STOCK_PRICES (
    price_id STRING PRIMARY KEY,
    ticker STRING NOT NULL,
    date DATE NOT NULL,
    
    -- OHLCV Data
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close FLOAT,
    
    -- Metadata
    data_source STRING, -- 'yahoo_finance', 'alpha_vantage', etc.
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    UNIQUE (ticker, date)
);

-- Cluster by ticker and date for performance
ALTER TABLE STOCK_PRICES CLUSTER BY (ticker, date);

-- ============================================
-- 2. COMPANY FUNDAMENTALS
-- ============================================
CREATE OR REPLACE TABLE COMPANY_FUNDAMENTALS (
    fundamental_id STRING PRIMARY KEY,
    ticker STRING NOT NULL,
    fiscal_quarter STRING, -- e.g., 'Q1-2024'
    report_date DATE NOT NULL,
    
    -- Income Statement
    revenue FLOAT,
    gross_profit FLOAT,
    operating_income FLOAT,
    net_income FLOAT,
    earnings_per_share FLOAT,
    
    -- Balance Sheet
    total_assets FLOAT,
    total_liabilities FLOAT,
    stockholders_equity FLOAT,
    cash_and_equivalents FLOAT,
    total_debt FLOAT,
    
    -- Cash Flow
    operating_cash_flow FLOAT,
    investing_cash_flow FLOAT,
    financing_cash_flow FLOAT,
    free_cash_flow FLOAT,
    
    -- Ratios
    pe_ratio FLOAT,
    pb_ratio FLOAT,
    debt_to_equity FLOAT,
    current_ratio FLOAT,
    quick_ratio FLOAT,
    roe FLOAT, -- Return on Equity
    roa FLOAT, -- Return on Assets
    
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================
-- 3. COMPANY PROFILES
-- ============================================
CREATE OR REPLACE TABLE COMPANY_PROFILES (
    ticker STRING PRIMARY KEY,
    company_name STRING NOT NULL,
    sector STRING,
    industry STRING,
    market_cap FLOAT,
    country STRING,
    exchange STRING,
    currency STRING,
    website STRING,
    description TEXT,
    employees INT,
    founded_year INT,
    ceo STRING,
    
    -- Updated info
    last_updated TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================
-- 4. SECTOR & INDUSTRY MAPPINGS
-- ============================================
CREATE OR REPLACE TABLE SECTOR_INDUSTRY_MAP (
    ticker STRING PRIMARY KEY,
    sector STRING NOT NULL,
    industry STRING NOT NULL,
    sub_industry STRING,
    gics_code STRING, -- Global Industry Classification Standard
    
    FOREIGN KEY (ticker) REFERENCES COMPANY_PROFILES(ticker)
);

-- ============================================
-- 5. TECHNICAL INDICATORS (Pre-calculated)
-- ============================================
CREATE OR REPLACE TABLE TECHNICAL_INDICATORS (
    indicator_id STRING PRIMARY KEY,
    ticker STRING NOT NULL,
    date DATE NOT NULL,
    
    -- Moving Averages
    sma_20 FLOAT,
    sma_50 FLOAT,
    sma_200 FLOAT,
    ema_20 FLOAT,
    ema_50 FLOAT,
    
    -- Momentum Indicators
    rsi_14 FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_histogram FLOAT,
    
    -- Volatility Indicators
    bollinger_upper FLOAT,
    bollinger_middle FLOAT,
    bollinger_lower FLOAT,
    atr_14 FLOAT, -- Average True Range
    
    -- Volume Indicators
    volume_sma_20 FLOAT,
    obv FLOAT, -- On-Balance Volume
    
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    UNIQUE (ticker, date)
);

-- ============================================
-- 6. MARKET INDICES
-- ============================================
CREATE OR REPLACE TABLE MARKET_INDICES (
    index_id STRING PRIMARY KEY,
    index_name STRING NOT NULL, -- 'S&P 500', 'NASDAQ', 'DOW'
    ticker STRING, -- '^GSPC', '^IXIC', '^DJI'
    date DATE NOT NULL,
    
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    
    -- Calculated metrics
    daily_return FLOAT,
    volatility_30d FLOAT,
    
    UNIQUE (index_name, date)
);

-- ============================================
-- 7. DIVIDEND HISTORY
-- ============================================
CREATE OR REPLACE TABLE DIVIDEND_HISTORY (
    dividend_id STRING PRIMARY KEY,
    ticker STRING NOT NULL,
    ex_date DATE NOT NULL,
    payment_date DATE,
    record_date DATE,
    dividend_amount FLOAT NOT NULL,
    dividend_type STRING, -- 'regular', 'special'
    
    UNIQUE (ticker, ex_date)
);

-- ============================================
-- 8. STOCK SPLITS
-- ============================================
CREATE OR REPLACE TABLE STOCK_SPLITS (
    split_id STRING PRIMARY KEY,
    ticker STRING NOT NULL,
    split_date DATE NOT NULL,
    split_ratio STRING, -- e.g., '2:1', '3:2'
    split_factor FLOAT, -- e.g., 2.0, 1.5
    
    UNIQUE (ticker, split_date)
);

-- ============================================
-- 9. NEWS & SENTIMENT (Optional)
-- ============================================
CREATE OR REPLACE TABLE NEWS_SENTIMENT (
    news_id STRING PRIMARY KEY,
    ticker STRING,
    published_date TIMESTAMP_NTZ,
    headline STRING,
    summary TEXT,
    source STRING,
    url STRING,
    
    -- Sentiment Analysis
    sentiment_score FLOAT, -- -1 to 1
    sentiment_label STRING, -- 'positive', 'neutral', 'negative'
    confidence FLOAT,
    
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================
-- 10. CORRELATION MATRIX (Pre-calculated)
-- ============================================
CREATE OR REPLACE TABLE CORRELATION_MATRIX (
    correlation_id STRING PRIMARY KEY,
    ticker_1 STRING NOT NULL,
    ticker_2 STRING NOT NULL,
    calculation_date DATE NOT NULL,
    lookback_days INT, -- e.g., 30, 90, 252
    correlation FLOAT NOT NULL,
    
    UNIQUE (ticker_1, ticker_2, calculation_date, lookback_days)
);

-- ============================================
-- INDEXES for Performance
-- ============================================
CREATE INDEX idx_stock_prices_ticker_date ON STOCK_PRICES(ticker, date);
CREATE INDEX idx_fundamentals_ticker_date ON COMPANY_FUNDAMENTALS(ticker, report_date);
CREATE INDEX idx_technical_ticker_date ON TECHNICAL_INDICATORS(ticker, date);
CREATE INDEX idx_market_indices_date ON MARKET_INDICES(index_name, date);
CREATE INDEX idx_news_ticker_date ON NEWS_SENTIMENT(ticker, published_date);

-- ============================================
-- MATERIALIZED VIEWS for Fast Access
-- ============================================

-- Latest stock prices
CREATE OR REPLACE VIEW VW_LATEST_STOCK_PRICES AS
SELECT 
    ticker,
    date,
    close as current_price,
    volume,
    LAG(close) OVER (PARTITION BY ticker ORDER BY date) as prev_close,
    ((close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) 
     / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) * 100 as daily_return_pct
FROM STOCK_PRICES
WHERE date = (SELECT MAX(date) FROM STOCK_PRICES WHERE ticker = STOCK_PRICES.ticker);

-- Latest company fundamentals
CREATE OR REPLACE VIEW VW_LATEST_FUNDAMENTALS AS
SELECT 
    cp.ticker,
    cp.company_name,
    cp.sector,
    cp.industry,
    cf.*
FROM COMPANY_PROFILES cp
JOIN COMPANY_FUNDAMENTALS cf ON cp.ticker = cf.ticker
WHERE cf.report_date = (
    SELECT MAX(report_date) 
    FROM COMPANY_FUNDAMENTALS 
    WHERE ticker = cp.ticker
);

-- Stock with indicators
CREATE OR REPLACE VIEW VW_STOCKS_WITH_INDICATORS AS
SELECT 
    sp.ticker,
    sp.date,
    sp.close,
    sp.volume,
    ti.rsi_14,
    ti.macd,
    ti.sma_20,
    ti.sma_50,
    ti.sma_200,
    ti.bollinger_upper,
    ti.bollinger_lower
FROM STOCK_PRICES sp
LEFT JOIN TECHNICAL_INDICATORS ti ON sp.ticker = ti.ticker AND sp.date = ti.date;

-- Sector performance
CREATE OR REPLACE VIEW VW_SECTOR_PERFORMANCE AS
SELECT 
    cp.sector,
    COUNT(DISTINCT sp.ticker) as num_stocks,
    AVG(sp.daily_return_pct) as avg_daily_return,
    AVG(cf.pe_ratio) as avg_pe_ratio,
    AVG(cf.roe) as avg_roe,
    SUM(cp.market_cap) as total_market_cap
FROM COMPANY_PROFILES cp
JOIN VW_LATEST_STOCK_PRICES sp ON cp.ticker = sp.ticker
LEFT JOIN VW_LATEST_FUNDAMENTALS cf ON cp.ticker = cf.ticker
GROUP BY cp.sector;