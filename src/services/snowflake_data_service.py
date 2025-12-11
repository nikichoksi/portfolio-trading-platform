import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.database.snowflake_config import SnowflakeConfig
import logging

logger = logging.getLogger(__name__)

class SnowflakeDataService:
    """Service to query market data from Snowflake"""
    
    def __init__(self):
        self.config = SnowflakeConfig()
        self.engine = self.config.get_sqlalchemy_engine()
    
    # ============================================
    # Stock Price Queries
    # ============================================
    
    def get_historical_prices(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
        days: int = 365
    ) -> pd.DataFrame:
        """Get historical prices for tickers"""
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT 
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                adjusted_close
            FROM STOCK_PRICES
            WHERE ticker IN ('{tickers_str}')
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY ticker, date
        """
        
        df = pd.read_sql(query, self.engine)
        logger.info(f"✅ Retrieved {len(df)} price records from Snowflake")
        return df
    
    def get_latest_prices(self, tickers: List[str]) -> pd.DataFrame:
        """Get latest prices for tickers"""
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT * FROM VW_LATEST_STOCK_PRICES
            WHERE ticker IN ('{tickers_str}')
        """
        
        return pd.read_sql(query, self.engine)
    
    # ============================================
    # Company Data Queries
    # ============================================
    
    def get_company_profiles(self, tickers: List[str]) -> pd.DataFrame:
        """Get company profile information"""
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT * FROM COMPANY_PROFILES
            WHERE ticker IN ('{tickers_str}')
        """
        
        return pd.read_sql(query, self.engine)
    
    def get_sector_allocation(self, tickers: List[str]) -> Dict[str, float]:
        """Get sector allocation for portfolio"""
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT 
                sector,
                COUNT(*) as count
            FROM COMPANY_PROFILES
            WHERE ticker IN ('{tickers_str}')
            AND sector IS NOT NULL
            GROUP BY sector
        """
        
        df = pd.read_sql(query, self.engine)
        total = df['count'].sum()
        
        return {row['sector']: row['count'] / total for _, row in df.iterrows()}
    
    # ============================================
    # Fundamental Data Queries
    # ============================================
    
    def get_latest_fundamentals(self, tickers: List[str]) -> pd.DataFrame:
        """Get latest fundamental data"""
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT * FROM VW_LATEST_FUNDAMENTALS
            WHERE ticker IN ('{tickers_str}')
        """
        
        return pd.read_sql(query, self.engine)
    
    # ============================================
    # Technical Indicator Queries
    # ============================================
    
    def get_technical_indicators(
        self,
        tickers: List[str],
        start_date: str = None,
        days: int = 90
    ) -> pd.DataFrame:
        """Get technical indicators"""
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT * FROM TECHNICAL_INDICATORS
            WHERE ticker IN ('{tickers_str}')
            AND date >= '{start_date}'
            ORDER BY ticker, date
        """
        
        return pd.read_sql(query, self.engine)
    
    def get_latest_indicators(self, tickers: List[str]) -> pd.DataFrame:
        """Get latest technical indicators"""
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT 
                ticker,
                date,
                rsi_14,
                macd,
                macd_signal,
                sma_20,
                sma_50,
                sma_200,
                bollinger_upper,
                bollinger_middle,
                bollinger_lower
            FROM TECHNICAL_INDICATORS
            WHERE ticker IN ('{tickers_str}')
            AND date = (
                SELECT MAX(date)
                FROM TECHNICAL_INDICATORS
                WHERE ticker = TECHNICAL_INDICATORS.ticker
            )
        """
        
        return pd.read_sql(query, self.engine)
    
    # ============================================
    # Market Benchmark Queries
    # ============================================
    
    def get_market_index_data(
        self,
        index_name: str = 'S&P 500',
        start_date: str = None,
        days: int = 365
    ) -> pd.DataFrame:
        """Get market index data for benchmark comparison"""
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        query = f"""
            SELECT 
                date,
                close,
                daily_return,
                volatility_30d
            FROM MARKET_INDICES
            WHERE index_name = '{index_name}'
            AND date >= '{start_date}'
            ORDER BY date
        """
        
        return pd.read_sql(query, self.engine)
    
    # ============================================
    # Correlation Queries
    # ============================================
    
    def get_correlation_matrix(
        self,
        tickers: List[str],
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """Get pre-calculated correlation matrix"""
        
        tickers_str = "','".join(tickers)
        
        query = f"""
            SELECT 
                ticker_1,
                ticker_2,
                correlation
            FROM CORRELATION_MATRIX
            WHERE ticker_1 IN ('{tickers_str}')
            AND ticker_2 IN ('{tickers_str}')
            AND lookback_days = {lookback_days}
            AND calculation_date = (SELECT MAX(calculation_date) FROM CORRELATION_MATRIX)
        """
        
        df = pd.read_sql(query, self.engine)
        
        # Pivot to matrix format
        if not df.empty:
            matrix = df.pivot(index='ticker_1', columns='ticker_2', values='correlation')
            return matrix
        
        return pd.DataFrame()
    
    # ============================================
    # Combined Query for Portfolio Analysis
    # ============================================
    
    def get_portfolio_data(
        self,
        tickers: List[str],
        days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """Get all data needed for portfolio analysis"""
        
        logger.info(f"📥 Fetching portfolio data for {len(tickers)} tickers from Snowflake...")
        
        return {
            'prices': self.get_historical_prices(tickers, days=days),
            'latest_prices': self.get_latest_prices(tickers),
            'profiles': self.get_company_profiles(tickers),
            'fundamentals': self.get_latest_fundamentals(tickers),
            'indicators': self.get_latest_indicators(tickers),
            'sector_allocation': self.get_sector_allocation(tickers),
            'sp500_data': self.get_market_index_data('S&P 500', days=days)
        }