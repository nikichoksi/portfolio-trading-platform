# src/database/snowflake_connector.py
"""
Snowflake connector for portfolio management.
Provides connection management and query execution for Snowflake data warehouse.
"""

import snowflake.connector
from snowflake.connector import DictCursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowflakeConfig:
    """Snowflake connection configuration"""
    
    def __init__(self):
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.database = os.getenv("SNOWFLAKE_DATABASE", "PORTFOLIO_DB")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self.role = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
        
        # Validate required configs
        if not all([self.account, self.user, self.password]):
            raise ValueError("Missing required Snowflake credentials in .env file")
    
    def get_connection_params(self) -> Dict[str, str]:
        """Get connection parameters as dictionary"""
        return {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
            "role": self.role
        }


class SnowflakeConnector:
    """Manages Snowflake database connections and operations"""
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        """
        Initialize Snowflake connector.
        
        Args:
            config: SnowflakeConfig object. If None, creates from environment.
        """
        self.config = config or SnowflakeConfig()
        self._connection = None
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for Snowflake connections.
        Ensures proper connection cleanup.
        
        Usage:
            with connector.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
        """
        conn = None
        try:
            conn = snowflake.connector.connect(
                **self.config.get_connection_params()
            )
            yield conn
        except Exception as e:
            logger.error(f"Snowflake connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            fetch: Whether to fetch results
        
        Returns:
            List of dictionaries (rows) if fetch=True, None otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(DictCursor)
            
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch:
                    results = cursor.fetchall()
                    return results
                else:
                    conn.commit()
                    return None
                    
            except Exception as e:
                logger.error(f"Query execution error: {str(e)}")
                logger.error(f"Query: {query}")
                raise
            finally:
                cursor.close()
    
    def execute_many(
        self,
        query: str,
        data: List[tuple]
    ) -> int:
        """
        Execute batch insert/update operations.
        
        Args:
            query: SQL query with placeholders
            data: List of tuples containing values
        
        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.executemany(query, data)
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                logger.error(f"Batch execution error: {str(e)}")
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def test_connection(self) -> bool:
        """
        Test Snowflake connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = self.execute_query("SELECT CURRENT_VERSION()")
            if result:
                logger.info(f"Snowflake connection successful. Version: {result[0]}")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def create_database_schema(self):
        """Create initial database schema for portfolio management"""
        
        schema_sql = """
        -- Create database and schema if not exists
        CREATE DATABASE IF NOT EXISTS PORTFOLIO_DB;
        USE DATABASE PORTFOLIO_DB;
        CREATE SCHEMA IF NOT EXISTS PUBLIC;
        USE SCHEMA PUBLIC;
        
        -- Stocks table
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL UNIQUE,
            company_name VARCHAR(255) NOT NULL,
            sector VARCHAR(100),
            industry VARCHAR(100),
            market_cap FLOAT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (id)
        );
        
        -- Transactions table
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL,
            transaction_type VARCHAR(4) NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
            quantity FLOAT NOT NULL,
            price FLOAT NOT NULL,
            transaction_date DATE NOT NULL,
            notes TEXT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (id),
            FOREIGN KEY (ticker) REFERENCES stocks(ticker)
        );
        
        -- Pending orders table
        CREATE TABLE IF NOT EXISTS pending_orders (
            id INTEGER AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL,
            order_type VARCHAR(4) NOT NULL CHECK (order_type IN ('BUY', 'SELL')),
            quantity FLOAT NOT NULL,
            limit_price FLOAT,
            order_status VARCHAR(20) NOT NULL CHECK (order_status IN ('PENDING', 'FILLED', 'CANCELLED')),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (id),
            FOREIGN KEY (ticker) REFERENCES stocks(ticker)
        );
        
        -- Market data table (for historical prices)
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open_price FLOAT,
            high_price FLOAT,
            low_price FLOAT,
            close_price FLOAT,
            volume BIGINT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (id),
            UNIQUE (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks(ticker)
        );
        
        -- Portfolio snapshots (for historical tracking)
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER AUTOINCREMENT,
            snapshot_date DATE NOT NULL,
            total_value FLOAT NOT NULL,
            total_cost_basis FLOAT NOT NULL,
            total_pnl FLOAT NOT NULL,
            total_pnl_pct FLOAT NOT NULL,
            num_positions INTEGER NOT NULL,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (id)
        );
        
        -- AI insights table (store agent analysis results)
        CREATE TABLE IF NOT EXISTS ai_insights (
            id INTEGER AUTOINCREMENT,
            insight_type VARCHAR(50) NOT NULL,
            portfolio_snapshot JSON,
            analysis_result JSON,
            agent_name VARCHAR(100),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (id)
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker);
        CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
        CREATE INDEX IF NOT EXISTS idx_orders_ticker ON pending_orders(ticker);
        CREATE INDEX IF NOT EXISTS idx_orders_status ON pending_orders(order_status);
        CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker, date);
        CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date);
        """
        
        # Execute schema creation
        for statement in schema_sql.split(';'):
            if statement.strip():
                try:
                    self.execute_query(statement, fetch=False)
                except Exception as e:
                    logger.warning(f"Schema creation warning: {str(e)}")
        
        logger.info("Database schema created successfully")


# Singleton instance
_snowflake_connector = None

def get_snowflake_connector() -> SnowflakeConnector:
    """Get singleton Snowflake connector instance"""
    global _snowflake_connector
    if _snowflake_connector is None:
        _snowflake_connector = SnowflakeConnector()
    return _snowflake_connector