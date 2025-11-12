#!/usr/bin/env python
"""
Standalone Market Data Ingestion Script
Runs the market data ingestion pipeline without Airflow.

This script performs the same tasks as the market_data_ingestion DAG:
1. Fetch portfolio tickers from SQLite
2. Fetch market data from Yahoo Finance
3. Load data into Snowflake
4. Validate data quality

Usage:
    python scripts/run_market_data_ingestion.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dotenv import load_dotenv
import yfinance as yf
from services.portfolio_service import PortfolioService
from data_warehouse.snowflake_connector import SnowflakeWarehouse


def fetch_portfolio_tickers():
    """Fetch all tickers from the portfolio."""
    print("\n[1/4] Fetching portfolio tickers...")
    service = PortfolioService()
    summary = service.get_portfolio_summary()

    if not summary or not summary.get('positions'):
        print("⚠ No positions found in portfolio")
        return []

    tickers = [pos.ticker for pos in summary['positions']]
    print(f"✓ Found {len(tickers)} tickers: {', '.join(tickers)}")
    return tickers


def fetch_market_indices():
    """Define market indices to track."""
    print("\n[2/4] Defining market indices...")
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000'
    }
    print(f"✓ Tracking {len(indices)} indices: {', '.join(indices.values())}")
    return list(indices.keys())


def fetch_stock_prices(tickers, lookback_days=30):
    """
    Fetch stock prices from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        lookback_days: Number of days to fetch

    Returns:
        Dict of ticker -> DataFrame
    """
    print(f"\n[3/4] Fetching stock prices (last {lookback_days} days)...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    price_data = {}
    successful = 0
    failed = 0

    for ticker in tickers:
        try:
            print(f"  Fetching {ticker}...", end=' ')
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if not hist.empty:
                price_data[ticker] = hist
                successful += 1
                print(f"✓ ({len(hist)} rows)")
            else:
                failed += 1
                print("✗ (no data)")

        except Exception as e:
            failed += 1
            print(f"✗ ({str(e)[:50]})")

    print(f"\n✓ Fetched data for {successful}/{len(tickers)} tickers")
    if failed > 0:
        print(f"⚠ Failed to fetch {failed} tickers")

    return price_data


def load_to_snowflake(price_data):
    """
    Load price data into Snowflake.

    Args:
        price_data: Dict of ticker -> DataFrame
    """
    print("\n[4/4] Loading data to Snowflake...")

    warehouse = SnowflakeWarehouse()

    try:
        # Ensure schema exists
        warehouse.setup_schema()

        # Insert data for each ticker
        successful = 0
        failed = 0

        for ticker, df in price_data.items():
            try:
                print(f"  Loading {ticker}...", end=' ')
                warehouse.insert_stock_prices(ticker, df)
                successful += 1
                print(f"✓ ({len(df)} rows)")
            except Exception as e:
                failed += 1
                print(f"✗ ({str(e)[:50]})")

        print(f"\n✓ Loaded {successful}/{len(price_data)} tickers to Snowflake")
        if failed > 0:
            print(f"⚠ Failed to load {failed} tickers")

    except Exception as e:
        print(f"✗ Error connecting to Snowflake: {e}")
        raise

    finally:
        warehouse.close()


def validate_data_quality(tickers):
    """
    Validate data quality in Snowflake.

    Args:
        tickers: List of ticker symbols to validate
    """
    print("\n[5/5] Validating data quality...")

    warehouse = SnowflakeWarehouse()

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        issues = []

        for ticker in tickers:
            # Check if we have recent data
            df = warehouse.get_stock_prices(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if df.empty:
                issues.append(f"{ticker}: No data in last 7 days")
            elif len(df) < 3:
                issues.append(f"{ticker}: Only {len(df)} records in last 7 days (expected ~5-7)")

        if issues:
            print(f"⚠ Data quality issues detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"✓ Data quality validated for {len(tickers)} tickers")

    except Exception as e:
        print(f"⚠ Data quality check failed: {e}")

    finally:
        warehouse.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Market Data Ingestion Pipeline - Portfolio Trading Platform")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load environment variables
    load_dotenv()

    try:
        # Step 1: Fetch portfolio tickers
        portfolio_tickers = fetch_portfolio_tickers()

        # Step 2: Fetch market indices
        index_tickers = fetch_market_indices()

        # Combine all tickers
        all_tickers = list(set(portfolio_tickers + index_tickers))
        print(f"\n✓ Total tickers to process: {len(all_tickers)}")

        # Step 3: Fetch stock prices
        price_data = fetch_stock_prices(all_tickers, lookback_days=30)

        if not price_data:
            print("\n⚠ No price data fetched. Exiting.")
            return 1

        # Step 4: Load to Snowflake
        load_to_snowflake(price_data)

        # Step 5: Validate data quality
        validate_data_quality(all_tickers)

        print("\n" + "=" * 70)
        print("Pipeline Completed Successfully!")
        print("=" * 70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
