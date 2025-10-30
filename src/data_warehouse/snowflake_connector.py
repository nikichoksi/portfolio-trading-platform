"""
Snowflake Data Warehouse Connector.
Manages connections to Snowflake and provides data access methods.
"""

import os
import yaml
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

try:
    import snowflake.connector
    from snowflake.connector import DictCursor
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("Warning: snowflake-connector-python not installed. Run: pip install snowflake-connector-python")


class SnowflakeWarehouse:
    """Snowflake data warehouse manager for portfolio analytics"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Snowflake connection.

        Args:
            config_path: Path to snowflake_config.yaml (defaults to config/snowflake_config.yaml)
        """
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("snowflake-connector-python is not installed")

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "snowflake_config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sf_config = self.config['snowflake']
        self.connection = None

    def _get_connection(self):
        """Get or create Snowflake connection"""
        if self.connection is None or self.connection.is_closed():
            # Helper function to resolve environment variables from config
            def resolve_env_var(value):
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var_name = value[2:-1]
                    return os.getenv(env_var_name, '')
                return value

            # Resolve all config values from environment variables
            account = resolve_env_var(self.sf_config['account'])
            user = resolve_env_var(self.sf_config['user'])
            password = resolve_env_var(self.sf_config['password'])
            warehouse = resolve_env_var(self.sf_config['warehouse'])
            database = resolve_env_var(self.sf_config['database'])
            schema = resolve_env_var(self.sf_config['schema'])
            role = resolve_env_var(self.sf_config.get('role', 'ACCOUNTADMIN'))

            self.connection = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema,
                role=role
            )

        return self.connection

    def close(self):
        """Close Snowflake connection"""
        if self.connection and not self.connection.is_closed():
            self.connection.close()

    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a SELECT query and return results as DataFrame.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            DataFrame with query results
        """
        conn = self._get_connection()
        cursor = conn.cursor(DictCursor)

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Fetch all results
            results = cursor.fetchall()

            # Convert to DataFrame
            if results:
                df = pd.DataFrame(results)
            else:
                df = pd.DataFrame()

            return df

        finally:
            cursor.close()

    def execute_command(self, command: str, params: Optional[Dict] = None) -> int:
        """
        Execute an INSERT/UPDATE/DELETE command.

        Args:
            command: SQL command string
            params: Command parameters

        Returns:
            Number of rows affected
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if params:
                cursor.execute(command, params)
            else:
                cursor.execute(command)

            conn.commit()
            return cursor.rowcount

        finally:
            cursor.close()

    def setup_schema(self):
        """Create database, schema, and tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Create database
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.sf_config['database']}")

            # Create schema
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.sf_config['database']}.{self.sf_config['schema']}")

            # Use the schema
            cursor.execute(f"USE SCHEMA {self.sf_config['database']}.{self.sf_config['schema']}")

            # Create stock_prices table
            cursor.execute("""
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
            """)

            # Create portfolio_snapshots table
            cursor.execute("""
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
            """)

            # Create portfolio_positions table
            cursor.execute("""
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
            """)

            # Create agent_analysis table
            cursor.execute("""
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
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_prices_date
                ON stock_prices(price_date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date
                ON portfolio_snapshots(snapshot_date)
            """)

            conn.commit()
            print("Snowflake schema created successfully")

        except Exception as e:
            print(f"Error setting up schema: {e}")
            raise

        finally:
            cursor.close()

    # ============= STOCK PRICE OPERATIONS =============

    def insert_stock_prices(self, ticker: str, price_data: pd.DataFrame):
        """
        Insert stock price data into Snowflake.

        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        if price_data.empty:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Prepare data for insertion
            for idx, row in price_data.iterrows():
                cursor.execute("""
                    MERGE INTO stock_prices AS target
                    USING (SELECT
                        %(ticker)s AS ticker,
                        %(price_date)s AS price_date,
                        %(open_price)s AS open_price,
                        %(high_price)s AS high_price,
                        %(low_price)s AS low_price,
                        %(close_price)s AS close_price,
                        %(volume)s AS volume
                    ) AS source
                    ON target.ticker = source.ticker AND target.price_date = source.price_date
                    WHEN MATCHED THEN UPDATE SET
                        open_price = source.open_price,
                        high_price = source.high_price,
                        low_price = source.low_price,
                        close_price = source.close_price,
                        volume = source.volume
                    WHEN NOT MATCHED THEN INSERT (ticker, price_date, open_price, high_price, low_price, close_price, volume)
                    VALUES (source.ticker, source.price_date, source.open_price, source.high_price, source.low_price, source.close_price, source.volume)
                """, {
                    'ticker': ticker,
                    'price_date': idx.strftime('%Y-%m-%d'),
                    'open_price': float(row['Open']),
                    'high_price': float(row['High']),
                    'low_price': float(row['Low']),
                    'close_price': float(row['Close']),
                    'volume': int(row['Volume'])
                })

            conn.commit()

        finally:
            cursor.close()

    def get_stock_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve stock prices from Snowflake.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with price data
        """
        query = """
            SELECT
                price_date,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM stock_prices
            WHERE ticker = %(ticker)s
            AND price_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY price_date
        """

        df = self.execute_query(query, {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })

        if not df.empty:
            df.set_index('PRICE_DATE', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        return df

    # ============= PORTFOLIO SNAPSHOT OPERATIONS =============

    def insert_portfolio_snapshot(self, summary: Dict, positions: List):
        """
        Insert portfolio snapshot into Snowflake.

        Args:
            summary: Portfolio summary dict
            positions: List of Position objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Insert snapshot
            cursor.execute("""
                INSERT INTO portfolio_snapshots
                (snapshot_date, total_market_value, total_cost_basis, total_unrealized_pnl,
                 total_unrealized_pnl_pct, num_positions)
                VALUES (%(snapshot_date)s, %(total_market_value)s, %(total_cost_basis)s,
                        %(total_unrealized_pnl)s, %(total_unrealized_pnl_pct)s, %(num_positions)s)
            """, {
                'snapshot_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_market_value': summary['total_market_value'],
                'total_cost_basis': summary['total_cost_basis'],
                'total_unrealized_pnl': summary['total_unrealized_pnl'],
                'total_unrealized_pnl_pct': summary['total_unrealized_pnl_pct'],
                'num_positions': summary['num_positions']
            })

            # Get the snapshot_id
            snapshot_id = cursor.lastrowid

            # Insert positions
            for pos in positions:
                cursor.execute("""
                    INSERT INTO portfolio_positions
                    (snapshot_id, ticker, quantity, avg_cost, current_price, market_value,
                     unrealized_pnl, unrealized_pnl_pct)
                    VALUES (%(snapshot_id)s, %(ticker)s, %(quantity)s, %(avg_cost)s,
                            %(current_price)s, %(market_value)s, %(unrealized_pnl)s, %(unrealized_pnl_pct)s)
                """, {
                    'snapshot_id': snapshot_id,
                    'ticker': pos.ticker,
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                })

            conn.commit()

        finally:
            cursor.close()

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get portfolio performance history.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with historical portfolio metrics
        """
        query = """
            SELECT
                snapshot_date,
                total_market_value,
                total_cost_basis,
                total_unrealized_pnl,
                total_unrealized_pnl_pct,
                num_positions
            FROM portfolio_snapshots
            WHERE snapshot_date >= DATEADD(day, -%(days)s, CURRENT_DATE())
            ORDER BY snapshot_date
        """

        return self.execute_query(query, {'days': days})

    # ============= AGENT ANALYSIS OPERATIONS =============

    def insert_agent_analysis(self, agent_name: str, analysis_type: str,
                            portfolio_tickers: str, metrics: Dict, recommendations: List):
        """
        Store AI agent analysis results.

        Args:
            agent_name: Name of the AI agent
            analysis_type: Type of analysis performed
            portfolio_tickers: Comma-separated list of tickers
            metrics: Dictionary of calculated metrics
            recommendations: List of recommendations
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO agent_analysis
                (analysis_date, agent_name, portfolio_tickers, analysis_type, metrics, recommendations)
                VALUES (%(analysis_date)s, %(agent_name)s, %(portfolio_tickers)s, %(analysis_type)s,
                        PARSE_JSON(%(metrics)s), PARSE_JSON(%(recommendations)s))
            """, {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'agent_name': agent_name,
                'portfolio_tickers': portfolio_tickers,
                'analysis_type': analysis_type,
                'metrics': str(metrics),
                'recommendations': str(recommendations)
            })

            conn.commit()

        finally:
            cursor.close()
