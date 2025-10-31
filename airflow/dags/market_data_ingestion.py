"""
Airflow DAG for Market Data Ingestion.
Fetches stock prices from Yahoo Finance and loads into Snowflake.

Schedule: Hourly during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


def fetch_portfolio_tickers(**context):
    """
    Fetch all tickers from the portfolio database.

    Returns:
        List of ticker symbols
    """
    from database.models import PortfolioDatabase

    db = PortfolioDatabase()
    stocks = db.get_all_stocks()
    tickers = [stock.ticker for stock in stocks]

    print(f"Found {len(tickers)} tickers in portfolio: {tickers}")

    # Push to XCom for next task
    context['ti'].xcom_push(key='tickers', value=tickers)

    return tickers


def fetch_market_indices(**context):
    """
    Fetch major market indices (S&P 500, NASDAQ, etc.)

    Returns:
        List of index symbols
    """
    indices = ['^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX']  # S&P 500, NASDAQ, Dow, Russell 2000, VIX

    print(f"Fetching market indices: {indices}")
    context['ti'].xcom_push(key='indices', value=indices)

    return indices


def fetch_stock_prices(**context):
    """
    Fetch stock prices from Yahoo Finance for all portfolio tickers.
    """
    import yfinance as yf

    # Get tickers from previous task
    tickers = context['ti'].xcom_pull(key='tickers', task_ids='fetch_portfolio_tickers')

    if not tickers:
        print("No tickers found in portfolio")
        return

    print(f"Fetching prices for {len(tickers)} tickers...")

    # Fetch data for last 30 days (to catch up on any missing data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    price_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if not hist.empty:
                price_data[ticker] = hist
                print(f"✓ Fetched {len(hist)} days of data for {ticker}")
            else:
                print(f"✗ No data available for {ticker}")

        except Exception as e:
            print(f"✗ Error fetching {ticker}: {e}")

    # Push to XCom
    context['ti'].xcom_push(key='price_data', value=price_data)

    return len(price_data)


def load_to_snowflake(**context):
    """
    Load stock price data into Snowflake warehouse.
    """
    from data_warehouse.snowflake_connector import SnowflakeWarehouse

    # Get price data from previous task
    price_data = context['ti'].xcom_pull(key='price_data', task_ids='fetch_stock_prices')

    if not price_data:
        print("No price data to load")
        return

    print(f"Loading data for {len(price_data)} stocks into Snowflake...")

    # Connect to Snowflake
    warehouse = SnowflakeWarehouse()

    # Ensure schema exists
    try:
        warehouse.setup_schema()
        print("✓ Snowflake schema verified")
    except Exception as e:
        print(f"✗ Error setting up schema: {e}")
        return

    # Insert data for each stock
    success_count = 0
    error_count = 0

    for ticker, df in price_data.items():
        try:
            warehouse.insert_stock_prices(ticker, df)
            success_count += 1
            print(f"✓ Loaded {ticker} ({len(df)} rows)")
        except Exception as e:
            error_count += 1
            print(f"✗ Error loading {ticker}: {e}")

    warehouse.close()

    print(f"\nLoad complete: {success_count} succeeded, {error_count} failed")

    return {'success': success_count, 'errors': error_count}


def validate_data_quality(**context):
    """
    Validate data quality in Snowflake.
    Check for missing prices, stale data, etc.
    """
    from data_warehouse.snowflake_connector import SnowflakeWarehouse

    warehouse = SnowflakeWarehouse()

    try:
        # Check for stocks with no recent data (older than 7 days)
        query = """
            SELECT ticker, MAX(price_date) AS last_update
            FROM stock_prices
            GROUP BY ticker
            HAVING last_update < DATEADD(day, -7, CURRENT_DATE())
        """

        stale_data = warehouse.execute_query(query)

        if not stale_data.empty:
            print(f"⚠ Warning: {len(stale_data)} stocks have stale data:")
            print(stale_data)
        else:
            print("✓ All stock data is up to date")

        # Check for price anomalies (>50% daily change - likely data error)
        anomaly_query = """
            SELECT
                ticker,
                price_date,
                close_price,
                LAG(close_price) OVER (PARTITION BY ticker ORDER BY price_date) AS prev_close,
                ((close_price - LAG(close_price) OVER (PARTITION BY ticker ORDER BY price_date)) /
                 LAG(close_price) OVER (PARTITION BY ticker ORDER BY price_date) * 100) AS daily_change_pct
            FROM stock_prices
            WHERE price_date >= DATEADD(day, -7, CURRENT_DATE())
            QUALIFY ABS(daily_change_pct) > 50
        """

        anomalies = warehouse.execute_query(anomaly_query)

        if not anomalies.empty:
            print(f"⚠ Warning: {len(anomalies)} potential data anomalies detected:")
            print(anomalies)
        else:
            print("✓ No data anomalies detected")

        warehouse.close()

    except Exception as e:
        print(f"✗ Error during data validation: {e}")


# Default arguments for the DAG
default_args = {
    'owner': 'portfolio-platform',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'market_data_ingestion',
    default_args=default_args,
    description='Fetch stock prices from Yahoo Finance and load into Snowflake',
    schedule='0 9-16 * * 1-5',  # Hourly from 9 AM to 4 PM ET, Mon-Fri
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    tags=['market-data', 'snowflake', 'yahoo-finance'],
)

# Define tasks
task_fetch_tickers = PythonOperator(
    task_id='fetch_portfolio_tickers',
    python_callable=fetch_portfolio_tickers,
    dag=dag,
)

task_fetch_indices = PythonOperator(
    task_id='fetch_market_indices',
    python_callable=fetch_market_indices,
    dag=dag,
)

task_fetch_prices = PythonOperator(
    task_id='fetch_stock_prices',
    python_callable=fetch_stock_prices,
    dag=dag,
)

task_load_snowflake = PythonOperator(
    task_id='load_to_snowflake',
    python_callable=load_to_snowflake,
    dag=dag,
)

task_validate = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag,
)

# Define task dependencies
task_fetch_tickers >> task_fetch_prices >> task_load_snowflake >> task_validate
task_fetch_indices >> task_validate
