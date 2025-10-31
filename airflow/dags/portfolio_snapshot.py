"""
Airflow DAG for Daily Portfolio Snapshots.
Captures portfolio state and stores in Snowflake for historical tracking.

Schedule: Daily at 5:00 PM ET (after market close)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


def capture_portfolio_snapshot(**context):
    """
    Capture current portfolio state and push to XCom.

    Returns:
        Portfolio summary dict
    """
    from services.portfolio_service import PortfolioService

    # Get portfolio service
    service = PortfolioService()

    # Get current portfolio summary
    summary = service.get_portfolio_summary()

    print(f"Portfolio Summary:")
    print(f"  Total Value: ${summary['total_market_value']:,.2f}")
    print(f"  Total Cost: ${summary['total_cost_basis']:,.2f}")
    print(f"  P&L: ${summary['total_unrealized_pnl']:,.2f} ({summary['total_unrealized_pnl_pct']:.2f}%)")
    print(f"  Positions: {summary['num_positions']}")

    # Push to XCom
    context['ti'].xcom_push(key='summary', value=summary)
    context['ti'].xcom_push(key='positions', value=[
        {
            'ticker': p.ticker,
            'quantity': p.quantity,
            'avg_cost': p.avg_cost,
            'current_price': p.current_price,
            'market_value': p.market_value,
            'unrealized_pnl': p.unrealized_pnl,
            'unrealized_pnl_pct': p.unrealized_pnl_pct
        }
        for p in summary['positions']
    ])

    return summary


def save_to_snowflake(**context):
    """
    Save portfolio snapshot to Snowflake.
    """
    from data_warehouse.snowflake_connector import SnowflakeWarehouse
    from dataclasses import dataclass

    # Get data from previous task
    summary = context['ti'].xcom_pull(key='summary', task_ids='capture_snapshot')
    positions_data = context['ti'].xcom_pull(key='positions', task_ids='capture_snapshot')

    if not summary or not positions_data:
        print("⚠ No portfolio data to save")
        return

    # Reconstruct Position objects (simple dict representation)
    @dataclass
    class Position:
        ticker: str
        quantity: float
        avg_cost: float
        current_price: float
        market_value: float
        unrealized_pnl: float
        unrealized_pnl_pct: float

    positions = [Position(**p) for p in positions_data]

    # Connect to Snowflake
    warehouse = SnowflakeWarehouse()

    try:
        # Ensure schema exists
        warehouse.setup_schema()

        # Insert snapshot
        warehouse.insert_portfolio_snapshot(summary, positions)

        print(f"✓ Saved portfolio snapshot to Snowflake")
        print(f"  Total Value: ${summary['total_market_value']:,.2f}")
        print(f"  Positions: {len(positions)}")

    except Exception as e:
        print(f"✗ Error saving to Snowflake: {e}")
        raise

    finally:
        warehouse.close()

    return {'success': True, 'num_positions': len(positions)}


def calculate_daily_metrics(**context):
    """
    Calculate additional daily metrics and analytics.
    """
    summary = context['ti'].xcom_pull(key='summary', task_ids='capture_snapshot')

    if not summary:
        print("⚠ No data to calculate metrics")
        return

    # Calculate daily performance metrics
    metrics = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_value': summary['total_market_value'],
        'total_cost': summary['total_cost_basis'],
        'total_return_pct': summary['total_unrealized_pnl_pct'],
        'num_positions': summary['num_positions']
    }

    print(f"Daily Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    context['ti'].xcom_push(key='daily_metrics', value=metrics)

    return metrics


def check_portfolio_alerts(**context):
    """
    Check for portfolio alerts (e.g., significant losses, concentration risk).
    """
    summary = context['ti'].xcom_pull(key='summary', task_ids='capture_snapshot')
    positions_data = context['ti'].xcom_pull(key='positions', task_ids='capture_snapshot')

    if not summary or not positions_data:
        return

    alerts = []

    # Check for significant daily loss (>5%)
    if summary['total_unrealized_pnl_pct'] < -5:
        alerts.append({
            'type': 'LOSS_ALERT',
            'message': f"Portfolio down {summary['total_unrealized_pnl_pct']:.2f}%",
            'severity': 'HIGH'
        })

    # Check for concentration risk (single position > 40% of portfolio)
    total_value = summary['total_market_value']
    for pos in positions_data:
        weight_pct = (pos['market_value'] / total_value * 100) if total_value > 0 else 0
        if weight_pct > 40:
            alerts.append({
                'type': 'CONCENTRATION_RISK',
                'message': f"{pos['ticker']} represents {weight_pct:.1f}% of portfolio",
                'severity': 'MEDIUM'
            })

    # Check for large position losses (>10%)
    for pos in positions_data:
        if pos['unrealized_pnl_pct'] < -10:
            alerts.append({
                'type': 'POSITION_LOSS',
                'message': f"{pos['ticker']} down {pos['unrealized_pnl_pct']:.2f}%",
                'severity': 'MEDIUM'
            })

    if alerts:
        print(f"⚠ {len(alerts)} alerts detected:")
        for alert in alerts:
            print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
    else:
        print("✓ No alerts detected")

    context['ti'].xcom_push(key='alerts', value=alerts)

    return alerts


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
    'portfolio_snapshot',
    default_args=default_args,
    description='Daily portfolio snapshot and analytics',
    schedule='0 17 * * 1-5',  # 5 PM ET, Mon-Fri (after market close)
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    tags=['portfolio', 'snapshot', 'snowflake'],
)

# Define tasks
task_capture = PythonOperator(
    task_id='capture_snapshot',
    python_callable=capture_portfolio_snapshot,
    dag=dag,
)

task_save = PythonOperator(
    task_id='save_to_snowflake',
    python_callable=save_to_snowflake,
    dag=dag,
)

task_metrics = PythonOperator(
    task_id='calculate_daily_metrics',
    python_callable=calculate_daily_metrics,
    dag=dag,
)

task_alerts = PythonOperator(
    task_id='check_portfolio_alerts',
    python_callable=check_portfolio_alerts,
    dag=dag,
)

# Define task dependencies
task_capture >> [task_save, task_metrics, task_alerts]
