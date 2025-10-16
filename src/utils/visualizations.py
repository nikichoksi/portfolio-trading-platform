"""
Portfolio visualization utilities.
Creates interactive charts for portfolio analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
from ..core.portfolio_metrics import PortfolioAnalyzer


def create_performance_chart(
    prices: pd.DataFrame,
    holdings: Dict[str, float],
    title: str = "Portfolio Performance Over Time"
) -> go.Figure:
    """
    Create cumulative performance chart for portfolio and individual assets.

    Args:
        prices: DataFrame of historical prices
        holdings: Dictionary of {ticker: weight}
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Calculate returns
    returns = prices.pct_change().dropna()

    # Calculate portfolio returns
    weight_vector = np.array([holdings.get(col, 0) for col in returns.columns])
    portfolio_returns = returns.dot(weight_vector)

    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1

    fig = go.Figure()

    # Add portfolio line
    fig.add_trace(go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative * 100,
        mode='lines',
        name='Portfolio',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))

    # Add individual assets
    colors = px.colors.qualitative.Set2
    for idx, ticker in enumerate(holdings.keys()):
        if ticker in returns.columns:
            asset_cumulative = (1 + returns[ticker]).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=asset_cumulative.index,
                y=asset_cumulative * 100,
                mode='lines',
                name=ticker,
                line=dict(color=colors[idx % len(colors)], dash='dash'),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig


def create_drawdown_chart(
    returns: pd.Series,
    title: str = "Portfolio Drawdown Over Time"
) -> go.Figure:
    """
    Create drawdown chart showing peak-to-trough declines.

    Args:
        returns: Series of portfolio returns
        title: Chart title

    Returns:
        Plotly Figure object
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='red', width=2),
        name='Drawdown',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix"
) -> go.Figure:
    """
    Create correlation heatmap.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.around(correlation_matrix.values, decimals=2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


def create_sector_pie_chart(
    sector_exposure: Dict[str, float],
    title: str = "Sector Allocation"
) -> go.Figure:
    """
    Create sector allocation pie chart.

    Args:
        sector_exposure: Dictionary of {sector: weight}
        title: Chart title

    Returns:
        Plotly Figure object
    """
    sectors = list(sector_exposure.keys())
    weights = list(sector_exposure.values())

    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=weights,
        hole=0.4,
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='%{label}<br>%{percent}<extra></extra>',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        )
    )])

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=450
    )

    return fig


def create_risk_return_scatter(
    portfolio_metrics,
    benchmark_return: float = 0.10,
    benchmark_vol: float = 0.15,
    title: str = "Risk-Return Profile"
) -> go.Figure:
    """
    Create risk-return scatter plot.

    Args:
        portfolio_metrics: PortfolioMetrics object
        benchmark_return: Benchmark annual return
        benchmark_vol: Benchmark annual volatility
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add portfolio point
    fig.add_trace(go.Scatter(
        x=[portfolio_metrics.annual_volatility * 100],
        y=[portfolio_metrics.annual_return * 100],
        mode='markers+text',
        name='Your Portfolio',
        marker=dict(size=20, color='#3B82F6', symbol='star'),
        text=['Portfolio'],
        textposition='top center',
        hovertemplate='Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: ' +
                     f'{portfolio_metrics.sharpe_ratio:.3f}<extra></extra>'
    ))

    # Add benchmark point
    fig.add_trace(go.Scatter(
        x=[benchmark_vol * 100],
        y=[benchmark_return * 100],
        mode='markers+text',
        name='S&P 500',
        marker=dict(size=15, color='gray', symbol='diamond'),
        text=['S&P 500'],
        textposition='bottom center',
        hovertemplate='Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>'
    ))

    # Add efficient frontier guide line
    x_range = np.linspace(0, max(portfolio_metrics.annual_volatility * 100,
                                  benchmark_vol * 100) * 1.5, 100)
    y_range = x_range * (portfolio_metrics.annual_return /
                         portfolio_metrics.annual_volatility)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name='Sharpe Ratio Line',
        line=dict(color='lightgray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        template='plotly_white',
        height=500,
        hovermode='closest'
    )

    return fig


def create_rolling_metrics_chart(
    returns: pd.Series,
    window: int = 30,
    title: str = "Rolling Sharpe Ratio (30-day)"
) -> go.Figure:
    """
    Create rolling metrics chart (e.g., rolling Sharpe ratio).

    Args:
        returns: Series of returns
        window: Rolling window size
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Calculate rolling Sharpe ratio
    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        mode='lines',
        name='Rolling Sharpe',
        line=dict(color='#10B981', width=2),
        hovertemplate='%{y:.3f}<extra></extra>'
    ))

    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="Good (>1)")
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        template='plotly_white',
        height=400
    )

    return fig


def create_var_chart(
    returns: pd.Series,
    confidence_levels: List[float] = [0.90, 0.95, 0.99],
    title: str = "Value at Risk Distribution"
) -> go.Figure:
    """
    Create VaR visualization with return distribution.

    Args:
        returns: Series of returns
        confidence_levels: List of confidence levels for VaR
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Create histogram of returns
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Return Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))

    # Add VaR lines
    colors = ['orange', 'red', 'darkred']
    for conf, color in zip(confidence_levels, colors):
        var = -np.percentile(returns, (1 - conf) * 100) * 100
        fig.add_vline(
            x=-var,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=f"VaR {conf:.0%}: {var:.2f}%",
            annotation_position="top"
        )

    fig.update_layout(
        title=title,
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=450,
        showlegend=False,
        bargap=0.1
    )

    return fig
