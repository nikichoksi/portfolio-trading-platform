#!/usr/bin/env python
"""
Snowflake Setup Script
Initializes Snowflake database, schema, and tables for the portfolio platform.

Usage:
    python scripts/setup_snowflake.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dotenv import load_dotenv
from data_warehouse.snowflake_connector import SnowflakeWarehouse


def main():
    """Main setup function"""
    print("=" * 60)
    print("Snowflake Setup Script - Portfolio Trading Platform")
    print("=" * 60)

    # Load environment variables
    print("\n[1/4] Loading environment variables...")
    load_dotenv()
    print("✓ Environment variables loaded")

    # Connect to Snowflake
    print("\n[2/4] Connecting to Snowflake...")
    try:
        warehouse = SnowflakeWarehouse()
        conn = warehouse._get_connection()
        print(f"✓ Connected to Snowflake account: {conn.account}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nPlease check your credentials in .env file:")
        print("  - SNOWFLAKE_ACCOUNT")
        print("  - SNOWFLAKE_USER")
        print("  - SNOWFLAKE_PASSWORD")
        print("  - SNOWFLAKE_WAREHOUSE")
        return 1

    # Create schema
    print("\n[3/4] Creating database schema...")
    try:
        warehouse.setup_schema()
        print("✓ Database schema created successfully")
        print(f"  - Database: PORTFOLIO_DB")
        print(f"  - Schema: MARKET_DATA")
        print(f"  - Tables:")
        print(f"    • stock_prices")
        print(f"    • portfolio_snapshots")
        print(f"    • portfolio_positions")
        print(f"    • agent_analysis")
        print(f"    • market_indices")
        print(f"    • sector_performance")
    except Exception as e:
        print(f"✗ Schema creation failed: {e}")
        warehouse.close()
        return 1

    # Test the connection
    print("\n[4/4] Testing connection...")
    try:
        # Run a simple test query
        result = warehouse.execute_query("SELECT CURRENT_TIMESTAMP() AS now")
        if not result.empty:
            print(f"✓ Connection test successful")
            print(f"  Server time: {result.iloc[0]['NOW']}")
        else:
            print("⚠ Connection test returned no results")
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        warehouse.close()
        return 1

    # Close connection
    warehouse.close()

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the market data ingestion DAG to populate stock prices")
    print("   $ airflow dags trigger market_data_ingestion")
    print("\n2. Or manually insert data using Python:")
    print("   >>> from src.data_warehouse.snowflake_connector import SnowflakeWarehouse")
    print("   >>> import yfinance as yf")
    print("   >>> warehouse = SnowflakeWarehouse()")
    print("   >>> stock = yf.Ticker('AAPL')")
    print("   >>> warehouse.insert_stock_prices('AAPL', stock.history(period='1mo'))")
    print("\n3. Query your data:")
    print("   >>> df = warehouse.get_stock_prices('AAPL', '2024-01-01', '2024-01-31')")
    print("   >>> print(df)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
