import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import uuid
from src.database.snowflake_config import SnowflakeConfig
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeDataLoader:
    """ETL pipeline to load market data into Snowflake"""
    
    def __init__(self):
        self.config = SnowflakeConfig()
        self.conn = self.config.get_connection()
    
    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    # ============================================
    # Load Historical Prices
    # ============================================
    
    def load_historical_prices(
        self, 
        tickers: List[str], 
        start_date: str = None,
        end_date: str = None,
        period: str = '1y'
    ):
        """Load historical stock prices into Snowflake"""
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor = self.conn.cursor()
        
        for ticker in tickers:
            try:
                logger.info(f"📥 Loading prices for {ticker}...")
                
                # Download data from Yahoo Finance
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"⚠️ No data for {ticker}")
                    continue
                
                # Insert into Snowflake
                for index, row in df.iterrows():
                    price_id = str(uuid.uuid4())
                    
                    cursor.execute("""
                        MERGE INTO STOCK_PRICES AS target
                        USING (SELECT %s as ticker, %s as date) AS source
                        ON target.ticker = source.ticker AND target.date = source.date
                        WHEN MATCHED THEN
                            UPDATE SET 
                                open = %s,
                                high = %s,
                                low = %s,
                                close = %s,
                                volume = %s,
                                adjusted_close = %s
                        WHEN NOT MATCHED THEN
                            INSERT (price_id, ticker, date, open, high, low, close, volume, adjusted_close, data_source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        ticker, index.date(),
                        row['Open'], row['High'], row['Low'], row['Close'], 
                        row['Volume'], row['Close'],
                        price_id, ticker, index.date(),
                        row['Open'], row['High'], row['Low'], row['Close'],
                        row['Volume'], row['Close'], 'yahoo_finance'
                    ))
                
                self.conn.commit()
                logger.info(f"✅ Loaded {len(df)} records for {ticker}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {ticker}: {e}")
                self.conn.rollback()
        
        cursor.close()
    
    # ============================================
    # Load Company Profiles
    # ============================================
    
    def load_company_profiles(self, tickers: List[str]):
        """Load company profile information"""
        cursor = self.conn.cursor()
        
        for ticker in tickers:
            try:
                logger.info(f"📥 Loading profile for {ticker}...")
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                cursor.execute("""
                    MERGE INTO COMPANY_PROFILES AS target
                    USING (SELECT %s as ticker) AS source
                    ON target.ticker = source.ticker
                    WHEN MATCHED THEN
                        UPDATE SET
                            company_name = %s,
                            sector = %s,
                            industry = %s,
                            market_cap = %s,
                            country = %s,
                            exchange = %s,
                            website = %s,
                            description = %s,
                            employees = %s,
                            last_updated = CURRENT_TIMESTAMP()
                    WHEN NOT MATCHED THEN
                        INSERT (ticker, company_name, sector, industry, market_cap, 
                                country, exchange, website, description, employees)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    ticker,
                    info.get('longName', ticker),
                    info.get('sector'),
                    info.get('industry'),
                    info.get('marketCap'),
                    info.get('country'),
                    info.get('exchange'),
                    info.get('website'),
                    info.get('longBusinessSummary'),
                    info.get('fullTimeEmployees'),
                    ticker,
                    info.get('longName', ticker),
                    info.get('sector'),
                    info.get('industry'),
                    info.get('marketCap'),
                    info.get('country'),
                    info.get('exchange'),
                    info.get('website'),
                    info.get('longBusinessSummary'),
                    info.get('fullTimeEmployees')
                ))
                
                self.conn.commit()
                logger.info(f"✅ Loaded profile for {ticker}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load profile for {ticker}: {e}")
                self.conn.rollback()
        
        cursor.close()
    
    # ============================================
    # Load Company Fundamentals
    # ============================================
    
    def load_fundamentals(self, tickers: List[str]):
        """Load company fundamental data"""
        cursor = self.conn.cursor()
        
        for ticker in tickers:
            try:
                logger.info(f"📥 Loading fundamentals for {ticker}...")
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                fundamental_id = str(uuid.uuid4())
                
                cursor.execute("""
                    INSERT INTO COMPANY_FUNDAMENTALS (
                        fundamental_id, ticker, report_date,
                        revenue, gross_profit, net_income, earnings_per_share,
                        total_assets, total_liabilities, stockholders_equity,
                        cash_and_equivalents, total_debt,
                        pe_ratio, pb_ratio, debt_to_equity, current_ratio,
                        roe, roa
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    fundamental_id,
                    ticker,
                    datetime.now().date(),
                    info.get('totalRevenue'),
                    info.get('grossProfits'),
                    info.get('netIncomeToCommon'),
                    info.get('trailingEps'),
                    info.get('totalAssets'),
                    info.get('totalLiabilities'),
                    info.get('stockholdersEquity'),
                    info.get('totalCash'),
                    info.get('totalDebt'),
                    info.get('trailingPE'),
                    info.get('priceToBook'),
                    info.get('debtToEquity'),
                    info.get('currentRatio'),
                    info.get('returnOnEquity'),
                    info.get('returnOnAssets')
                ))
                
                self.conn.commit()
                logger.info(f"✅ Loaded fundamentals for {ticker}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load fundamentals for {ticker}: {e}")
                self.conn.rollback()
        
        cursor.close()
    
    # ============================================
    # Calculate and Load Technical Indicators
    # ============================================
    
    def calculate_and_load_indicators(self, tickers: List[str]):
        """Calculate technical indicators and load into Snowflake"""
        cursor = self.conn.cursor()
        
        for ticker in tickers:
            try:
                logger.info(f"📊 Calculating indicators for {ticker}...")
                
                # Fetch price data from Snowflake
                query = f"""
                    SELECT date, close, volume
                    FROM STOCK_PRICES
                    WHERE ticker = '{ticker}'
                    ORDER BY date DESC
                    LIMIT 300
                """
                
                df = pd.read_sql(query, self.config.get_sqlalchemy_engine())
                df = df.sort_values('date')
                
                if len(df) < 50:
                    logger.warning(f"⚠️ Not enough data for {ticker}")
                    continue
                
                # Calculate indicators
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['sma_200'] = df['close'].rolling(window=200).mean()
                
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi_14'] = 100 - (100 / (1 + rs))
                
                # MACD
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                
                # Bollinger Bands
                df['bollinger_middle'] = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                df['bollinger_upper'] = df['bollinger_middle'] + (std * 2)
                df['bollinger_lower'] = df['bollinger_middle'] - (std * 2)
                
                # Load into Snowflake
                for index, row in df.dropna().iterrows():
                    indicator_id = str(uuid.uuid4())
                    
                    cursor.execute("""
                        MERGE INTO TECHNICAL_INDICATORS AS target
                        USING (SELECT %s as ticker, %s as date) AS source
                        ON target.ticker = source.ticker AND target.date = source.date
                        WHEN MATCHED THEN
                            UPDATE SET
                                sma_20 = %s, sma_50 = %s, sma_200 = %s,
                                rsi_14 = %s, macd = %s, macd_signal = %s,
                                macd_histogram = %s,
                                bollinger_upper = %s, bollinger_middle = %s,
                                bollinger_lower = %s
                        WHEN NOT MATCHED THEN
                            INSERT (indicator_id, ticker, date,
                                    sma_20, sma_50, sma_200, rsi_14,
                                    macd, macd_signal, macd_histogram,
                                    bollinger_upper, bollinger_middle, bollinger_lower)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        ticker, row['date'],
                        row['sma_20'], row['sma_50'], row['sma_200'],
                        row['rsi_14'], row['macd'], row['macd_signal'],
                        row['macd_histogram'],
                        row['bollinger_upper'], row['bollinger_middle'],
                        row['bollinger_lower'],
                        indicator_id, ticker, row['date'],
                        row['sma_20'], row['sma_50'], row['sma_200'],
                        row['rsi_14'], row['macd'], row['macd_signal'],
                        row['macd_histogram'],
                        row['bollinger_upper'], row['bollinger_middle'],
                        row['bollinger_lower']
                    ))
                
                self.conn.commit()
                logger.info(f"✅ Loaded indicators for {ticker}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load indicators for {ticker}: {e}")
                self.conn.rollback()
        
        cursor.close()
    
    # ============================================
    # Convenience Method: Load Everything
    # ============================================
    
    def load_all_data(self, tickers: List[str], period: str = '1y'):
        """Load all data types for given tickers"""
        logger.info(f"📥 Starting full data load for {len(tickers)} tickers...")
        
        self.load_company_profiles(tickers)
        self.load_historical_prices(tickers, period=period)
        self.load_fundamentals(tickers)
        self.calculate_and_load_indicators(tickers)
        
        logger.info("✅ Full data load completed!")