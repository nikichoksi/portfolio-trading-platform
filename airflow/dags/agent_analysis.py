"""
Airflow DAG for AI Agent Portfolio Analysis.
Runs portfolio analysis agents and stores results in Snowflake.

Schedule: Daily at 6:00 PM ET (after portfolio snapshot)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


def run_portfolio_insight_agent(**context):
    """
    Run Portfolio Insight Agent analysis on current portfolio.

    Returns:
        Analysis summary dict
    """
    from services.portfolio_service import PortfolioService
    from agents.portfolio_agent import PortfolioInsightAgent
    from utils.portfolio_analytics import PortfolioAnalyzer

    # Get portfolio service
    service = PortfolioService()

    # Get current positions
    positions = service.get_positions()

    if not positions:
        print("⚠ No positions found in portfolio")
        return None

    # Initialize analyzer and agent
    analyzer = PortfolioAnalyzer()
    agent = PortfolioInsightAgent(service=service)

    # Prepare portfolio string for analysis
    portfolio_str = ", ".join([
        f"{p.quantity} shares {p.ticker}"
        for p in positions
    ])

    print(f"Analyzing portfolio: {portfolio_str}")

    # Run analysis
    analysis_text, metrics, _ = agent.analyze_live_portfolio()

    if not metrics:
        print("⚠ Could not generate portfolio metrics")
        return None

    # Prepare metrics dict
    metrics_dict = {
        "total_return": float(metrics.total_return),
        "annualized_return": float(metrics.annualized_return),
        "volatility": float(metrics.volatility),
        "sharpe_ratio": float(metrics.sharpe_ratio),
        "beta": float(metrics.beta),
        "max_drawdown": float(metrics.max_drawdown),
        "diversification_score": float(metrics.diversification_score),
        "risk_level": metrics.risk_level
    }

    # Prepare recommendations based on analysis
    recommendations = []

    if metrics.diversification_score < 60:
        recommendations.append({
            "type": "diversification",
            "priority": "high",
            "message": f"Low diversification score ({metrics.diversification_score:.0f}/100). Consider adding positions to reduce concentration risk."
        })

    if metrics.sharpe_ratio < 1.0:
        recommendations.append({
            "type": "risk_adjusted_returns",
            "priority": "medium",
            "message": f"Sharpe ratio is {metrics.sharpe_ratio:.2f}. Portfolio could benefit from improved risk-return balance."
        })

    if metrics.max_drawdown < -20:
        recommendations.append({
            "type": "drawdown_risk",
            "priority": "high",
            "message": f"Maximum drawdown of {metrics.max_drawdown:.1f}% indicates high volatility. Consider defensive positions."
        })

    if metrics.volatility > 30:
        recommendations.append({
            "type": "volatility",
            "priority": "medium",
            "message": f"High volatility ({metrics.volatility:.1f}%). Review position sizing and sector concentration."
        })

    if not recommendations:
        recommendations.append({
            "type": "general",
            "priority": "low",
            "message": "Portfolio metrics are within acceptable ranges. Continue monitoring market conditions."
        })

    result = {
        "agent_name": "Portfolio Insight Agent",
        "analysis_type": "portfolio_analysis",
        "portfolio_tickers": ",".join([p.ticker for p in positions]),
        "metrics": metrics_dict,
        "recommendations": recommendations,
        "analysis_text": analysis_text
    }

    print(f"✓ Portfolio Insight Agent analysis complete")
    print(f"  Analyzed {len(positions)} positions")
    print(f"  Generated {len(recommendations)} recommendations")

    # Push to XCom
    context['ti'].xcom_push(key='portfolio_insight', value=result)

    return result


def run_risk_profiler_agent(**context):
    """
    Run Risk Profiler Agent analysis on current portfolio.

    Returns:
        Risk profile analysis dict
    """
    from services.portfolio_service import PortfolioService
    from agents.risk_profiler_agent import RiskProfilerAgent, RiskProfile, RiskProfiler
    from utils.portfolio_analytics import PortfolioAnalyzer

    # Get portfolio service
    service = PortfolioService()

    # Get current positions
    positions = service.get_positions()

    if not positions:
        print("⚠ No positions found in portfolio")
        return None

    # Initialize analyzer
    analyzer = PortfolioAnalyzer()
    profiler = RiskProfiler()

    # Get portfolio metrics
    tickers = [p.ticker for p in positions]
    price_history = {}
    sectors = {}

    for ticker in tickers:
        df = service.get_price_history(ticker, "3mo")
        if not df.empty:
            price_history[ticker] = df

        info = service.get_stock_info(ticker)
        if info:
            sectors[ticker] = info.get('sector', 'Other')

    metrics = analyzer.calculate_portfolio_metrics(positions, price_history, sectors)

    # Suggest best risk profile
    suggested_profile, reasoning = profiler.suggest_profile(metrics)

    # Evaluate fit for all profiles
    profile_scores = {}
    profile_evaluations = {}

    for profile in RiskProfile:
        evaluation = profiler.evaluate_profile_fit(metrics, profile)
        profile_scores[profile.value] = evaluation['fit_score']
        profile_evaluations[profile.value] = {
            "fit_score": evaluation['fit_score'],
            "mismatches": len(evaluation['mismatches']),
            "recommendation": evaluation['recommendation']
        }

    # Prepare metrics dict
    metrics_dict = {
        "suggested_profile": suggested_profile.value,
        "fit_score": profile_scores[suggested_profile.value],
        "profile_scores": profile_scores,
        "volatility": float(metrics.volatility),
        "beta": float(metrics.beta),
        "diversification_score": float(metrics.diversification_score),
        "reasoning": reasoning
    }

    # Generate recommendations
    recommendations = []

    # Evaluate against conservative profile
    conservative_eval = profiler.evaluate_profile_fit(metrics, RiskProfile.CONSERVATIVE)
    if conservative_eval['mismatches']:
        for mismatch in conservative_eval['mismatches']:
            if mismatch['severity'] == 'high':
                recommendations.append({
                    "type": "risk_alignment",
                    "priority": "high",
                    "message": f"Conservative investors: {mismatch['metric']} is {mismatch['current']} (target: {mismatch['target']})"
                })

    # Add profile-specific recommendations
    if suggested_profile == RiskProfile.AGGRESSIVE:
        recommendations.append({
            "type": "profile_match",
            "priority": "low",
            "message": f"Portfolio aligns with aggressive growth strategy. Monitor volatility during market stress."
        })
    elif suggested_profile == RiskProfile.CONSERVATIVE:
        recommendations.append({
            "type": "profile_match",
            "priority": "low",
            "message": f"Portfolio aligns with conservative strategy. Well-positioned for capital preservation."
        })
    else:
        recommendations.append({
            "type": "profile_match",
            "priority": "low",
            "message": f"Portfolio shows moderate risk profile with balanced growth and stability."
        })

    result = {
        "agent_name": "Risk Profiler Agent",
        "analysis_type": "risk_profiling",
        "portfolio_tickers": ",".join(tickers),
        "metrics": metrics_dict,
        "recommendations": recommendations
    }

    print(f"✓ Risk Profiler Agent analysis complete")
    print(f"  Suggested Profile: {suggested_profile.value}")
    print(f"  Fit Score: {profile_scores[suggested_profile.value]:.0f}/100")
    print(f"  Generated {len(recommendations)} recommendations")

    # Push to XCom
    context['ti'].xcom_push(key='risk_profiler', value=result)

    return result


def save_agent_analyses_to_snowflake(**context):
    """
    Save all agent analysis results to Snowflake.
    """
    from data_warehouse.snowflake_connector import SnowflakeWarehouse

    # Get analyses from previous tasks
    portfolio_insight = context['ti'].xcom_pull(key='portfolio_insight', task_ids='run_portfolio_insight')
    risk_profiler = context['ti'].xcom_pull(key='risk_profiler', task_ids='run_risk_profiler')

    analyses = [portfolio_insight, risk_profiler]

    # Filter out None values
    analyses = [a for a in analyses if a is not None]

    if not analyses:
        print("⚠ No analysis results to save")
        return

    # Connect to Snowflake
    warehouse = SnowflakeWarehouse()

    try:
        # Ensure schema exists
        warehouse.setup_schema()

        success_count = 0

        for analysis in analyses:
            try:
                warehouse.insert_agent_analysis(
                    agent_name=analysis['agent_name'],
                    analysis_type=analysis['analysis_type'],
                    portfolio_tickers=analysis['portfolio_tickers'],
                    metrics=analysis['metrics'],
                    recommendations=analysis['recommendations']
                )
                success_count += 1
                print(f"✓ Saved {analysis['agent_name']} analysis to Snowflake")
            except Exception as e:
                print(f"✗ Error saving {analysis['agent_name']}: {e}")

        print(f"\nLoad complete: {success_count}/{len(analyses)} analyses saved")

        return {'success': success_count, 'total': len(analyses)}

    except Exception as e:
        print(f"✗ Error saving to Snowflake: {e}")
        raise

    finally:
        warehouse.close()


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
    'agent_analysis',
    default_args=default_args,
    description='Run AI agent portfolio analysis and store results',
    schedule='0 18 * * 1-5',  # 6 PM ET, Mon-Fri (after portfolio snapshot at 5 PM)
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    tags=['portfolio', 'ai-agents', 'snowflake', 'analysis'],
)

# Define tasks
task_portfolio_insight = PythonOperator(
    task_id='run_portfolio_insight',
    python_callable=run_portfolio_insight_agent,
    dag=dag,
)

task_risk_profiler = PythonOperator(
    task_id='run_risk_profiler',
    python_callable=run_risk_profiler_agent,
    dag=dag,
)

task_save = PythonOperator(
    task_id='save_to_snowflake',
    python_callable=save_agent_analyses_to_snowflake,
    dag=dag,
)

# Define task dependencies
[task_portfolio_insight, task_risk_profiler] >> task_save
