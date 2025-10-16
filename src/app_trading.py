"""
Professional Trading Platform - Robinhood/Zerodha Style
Complete trading interface with market view, stock analytics, pattern detection, and order management.
"""

import streamlit as st
import os
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from services.portfolio_service import PortfolioService
from services.order_execution import OrderExecutionService, get_order_status_message
from database.models import PendingOrder
from utils.pattern_detection import PatternDetector, get_pattern_marker_style
from utils.market_data import get_all_market_stocks, get_popular_stocks, search_stocks
from agents.portfolio_agent import PortfolioInsightAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Portfolio Trading Platform",
    page_icon="Chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for trading platform look
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }

    /* Stock cards */
    .stock-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .stock-card:hover {
        border-color: #5d5fef;
        box-shadow: 0 2px 8px rgba(93,95,239,0.15);
    }

    /* Price displays */
    .price-up {
        color: #00C805;
        font-weight: 600;
    }
    .price-down {
        color: #FF5000;
        font-weight: 600;
    }

    /* Sector headers */
    .sector-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #5d5fef;
    }

    /* Order cards */
    .order-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 4px solid #5d5fef;
        margin-bottom: 0.5rem;
    }

    /* Buy/Sell buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }

    /* Pattern badges */
    .pattern-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .pattern-bullish {
        background: #d4edda;
        color: #155724;
    }
    .pattern-bearish {
        background: #f8d7da;
        color: #721c24;
    }
    .pattern-continuation {
        background: #d1ecf1;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if "service" not in st.session_state:
        st.session_state.service = PortfolioService()
    if "selected_stock" not in st.session_state:
        st.session_state.selected_stock = None
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "market"  # market, stock_detail, portfolio
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def create_candlestick_chart_with_patterns(df: pd.DataFrame, ticker: str, patterns: List) -> go.Figure:
    """Create candlestick chart with pattern markers"""
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#00C805',
        decreasing_line_color='#FF5000'
    ))

    # Add volume bars
    colors = ['#00C805' if close >= open else '#FF5000'
              for close, open in zip(df['Close'], df['Open'])]

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.3,
        yaxis='y2'
    ))

    # Add pattern markers
    for pattern in patterns:
        style = get_pattern_marker_style(pattern.pattern_type)

        for idx, price in pattern.marker_positions:
            if idx < len(df):
                fig.add_trace(go.Scatter(
                    x=[df.index[idx]],
                    y=[price],
                    mode='markers+text',
                    name=pattern.pattern_name,
                    marker=dict(
                        size=style['size'],
                        color=style['color'],
                        symbol=style['symbol'],
                        line=dict(color='white', width=2)
                    ),
                    text=[pattern.pattern_name[:3]],
                    textposition='top center',
                    showlegend=True,
                    hovertemplate=f'<b>{pattern.pattern_name}</b><br>' +
                                 f'{pattern.description}<br>' +
                                 f'Confidence: {pattern.confidence:.0%}<extra></extra>'
                ))

    fig.update_layout(
        title=f'{ticker} - Technical Analysis',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def render_market_view():
    """Render live market view grouped by sector"""
    st.markdown('<div class="main-header">Live Market</div>', unsafe_allow_html=True)

    service = st.session_state.service

    # Search bar
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("Search: Search stocks...", placeholder="Search by ticker or sector", key="market_search")
    with col2:
        view_mode = st.selectbox("View", ["Popular", "All Sectors"], key="market_view_mode")

    # Get market stocks
    if search_query:
        # Search mode
        results = search_stocks(search_query)
        if results:
            st.info(f"Found {len(results)} results")
            tickers_to_show = [ticker for ticker, sector in results]

            # Fetch prices
            with st.spinner("Fetching prices..."):
                live_prices = service.get_live_prices(tickers_to_show[:50])  # Limit to 50 for performance

            # Display results
            cols = st.columns(3)
            for idx, (ticker, sector) in enumerate(results[:50]):
                with cols[idx % 3]:
                    info = service.get_stock_info(ticker)
                    if info and info.get('current_price', 0) > 0:
                        render_stock_card_simple(ticker, sector, info)
        else:
            st.warning("No stocks found")
        return

    # Get all market stocks
    all_market_stocks = get_all_market_stocks()

    if view_mode == "Popular":
        # Show popular stocks
        st.subheader(" Most Popular Stocks")
        popular = get_popular_stocks(30)

        with st.spinner("Fetching live prices..."):
            live_prices = service.get_live_prices(popular)

        cols = st.columns(3)
        for idx, ticker in enumerate(popular):
            with cols[idx % 3]:
                info = service.get_stock_info(ticker)
                if info and info.get('current_price', 0) > 0:
                    from utils.market_data import get_stock_sector
                    sector = get_stock_sector(ticker)
                    render_stock_card_simple(ticker, sector, info)
    else:
        # Show all sectors
        for sector, tickers in all_market_stocks.items():
            with st.expander(f" {sector} ({len(tickers)} stocks)", expanded=False):
                # Fetch prices for this sector (in batches for performance)
                batch_size = 20
                for i in range(0, min(len(tickers), 60), batch_size):  # Show first 60 per sector
                    batch = tickers[i:i+batch_size]

                    with st.spinner(f"Loading {sector}..."):
                        live_prices = service.get_live_prices(batch)

                    cols = st.columns(4)
                    for idx, ticker in enumerate(batch):
                        with cols[idx % 4]:
                            info = service.get_stock_info(ticker)
                            if info and info.get('current_price', 0) > 0:
                                render_stock_card_simple(ticker, sector, info)


def render_stock_card_simple(ticker: str, sector: str, info: Dict):
    """Render simple stock card for market view"""
    current_price = info.get('current_price', 0)
    previous_close = info.get('previous_close', current_price)
    company_name = info.get('company_name', ticker)

    change = current_price - previous_close
    change_pct = (change / previous_close * 100) if previous_close > 0 else 0

    price_class = "price-up" if change >= 0 else "price-down"
    arrow = "▲" if change >= 0 else "▼"

    # Create compact card
    st.markdown(f"""
    <div style="background: white; padding: 0.8rem; border-radius: 6px; border: 1px solid #e0e0e0; margin-bottom: 0.5rem;">
        <div style="font-weight: 600; font-size: 0.95rem;">{ticker}</div>
        <div style="color: #666; font-size: 0.75rem; margin-bottom: 0.4rem;">{company_name[:25]}</div>
        <div style="font-size: 1.1rem; font-weight: 700;">${current_price:.2f}</div>
        <div class="{price_class}" style="font-size: 0.8rem;">
            {arrow} {abs(change_pct):.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View", key=f"view_{ticker}_{sector}", use_container_width=True, type="secondary"):
        st.session_state.selected_stock = ticker
        st.session_state.view_mode = "stock_detail"
        st.rerun()


def render_stock_card(data: Dict):
    """Render individual stock card"""
    stock = data['stock']
    info = data['info']
    current_price = info.get('current_price', 0)
    previous_close = info.get('previous_close', current_price)

    change = current_price - previous_close
    change_pct = (change / previous_close * 100) if previous_close > 0 else 0

    price_class = "price-up" if change >= 0 else "price-down"
    arrow = "▲" if change >= 0 else "▼"

    # Create card with button
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"""
        <div style="font-weight: 600; font-size: 1.1rem;">{stock.ticker}</div>
        <div style="color: #666; font-size: 0.9rem;">{stock.company_name[:30]}</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top: 0.5rem;">
            <span style="font-size: 1.3rem; font-weight: 700;">${current_price:.2f}</span>
            <span class="{price_class}" style="margin-left: 0.5rem; font-size: 0.95rem;">
                {arrow} ${abs(change):.2f} ({abs(change_pct):.2f}%)
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("View →", key=f"view_{stock.ticker}", use_container_width=True):
            st.session_state.selected_stock = stock.ticker
            st.session_state.view_mode = "stock_detail"
            st.rerun()

    st.divider()


def render_stock_detail():
    """Render detailed stock analysis page"""
    ticker = st.session_state.selected_stock
    service = st.session_state.service

    # Back button
    if st.button("← Back to Market"):
        st.session_state.view_mode = "market"
        st.session_state.selected_stock = None
        st.rerun()

    st.markdown(f'<div class="main-header">{ticker} - Stock Analysis</div>', unsafe_allow_html=True)

    # Get stock info
    info = service.get_stock_info(ticker)

    if not info or info.get('current_price', 0) == 0:
        st.error("Unable to fetch stock data")
        return

    # Header with key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    current_price = info['current_price']
    previous_close = info['previous_close']
    change = current_price - previous_close
    change_pct = (change / previous_close * 100) if previous_close > 0 else 0

    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
    with col2:
        st.metric("Day High", f"${info['day_high']:.2f}")
    with col3:
        st.metric("Day Low", f"${info['day_low']:.2f}")
    with col4:
        st.metric("52W High", f"${info['52_week_high']:.2f}")
    with col5:
        st.metric("52W Low", f"${info['52_week_low']:.2f}")

    st.divider()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Chart", "Trade", "Orders"])

    # TAB 1: CHART ANALYSIS
    with tab1:
        # Period selector
        col1, col2 = st.columns([1, 4])

        with col1:
            period = st.selectbox(
                "Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                format_func=lambda x: {
                    "1d": "1 Day",
                    "5d": "5 Days",
                    "1mo": "1 Month",
                    "3mo": "3 Months",
                    "6mo": "6 Months",
                    "1y": "1 Year",
                    "2y": "2 Years"
                }[x],
                index=4  # Default to 6 months
            )

        # Fetch price data
        with st.spinner("Loading chart data..."):
            price_history = service.get_price_history(ticker, period)

        if not price_history.empty:
            # Detect patterns
            detector = PatternDetector()
            all_patterns = detector.detect_all_patterns(price_history)

            # Filter and prioritize patterns
            # 1. Remove duplicate pattern types (keep highest confidence)
            # 2. Limit to top 5 most confident patterns
            # 3. Prioritize recent patterns (closer to end_idx)

            unique_patterns = {}
            for pattern in all_patterns:
                key = pattern.pattern_name
                if key not in unique_patterns or pattern.confidence > unique_patterns[key].confidence:
                    unique_patterns[key] = pattern

            # Sort by confidence and recency, take top 5
            filtered_patterns = sorted(
                unique_patterns.values(),
                key=lambda p: (p.confidence, p.end_idx),
                reverse=True
            )[:5]

            # Display detected patterns
            if filtered_patterns:
                st.subheader(" Key Patterns Detected")

                # Display in a grid layout with max 5 columns
                num_cols = min(len(filtered_patterns), 5)
                pattern_cols = st.columns(num_cols)

                for idx, pattern in enumerate(filtered_patterns):
                    with pattern_cols[idx]:
                        badge_class = f"pattern-{pattern.pattern_type}"
                        st.markdown(f"""
                        <div class="pattern-badge {badge_class}">
                            {pattern.pattern_name}<br/>
                            {pattern.confidence:.0%}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(pattern.description[:50] + "..." if len(pattern.description) > 50 else pattern.description)

            # Create and display chart (use all patterns for markers, but filtered for display)
            fig = create_candlestick_chart_with_patterns(price_history, ticker, filtered_patterns)
            st.plotly_chart(fig, use_container_width=True)

            # Technical indicators
            st.subheader("Technical Indicators")

            col1, col2, col3, col4 = st.columns(4)

            # Calculate simple indicators
            sma_20 = price_history['Close'].rolling(20).mean().iloc[-1] if len(price_history) >= 20 else 0
            sma_50 = price_history['Close'].rolling(50).mean().iloc[-1] if len(price_history) >= 50 else 0
            volatility = price_history['Close'].pct_change().std() * np.sqrt(252) * 100

            with col1:
                st.metric("20-Day SMA", f"${sma_20:.2f}" if sma_20 > 0 else "N/A")
            with col2:
                st.metric("50-Day SMA", f"${sma_50:.2f}" if sma_50 > 0 else "N/A")
            with col3:
                st.metric("Volatility", f"{volatility:.2f}%")
            with col4:
                avg_volume = price_history['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")

        else:
            st.warning("No price data available for this period")

    # TAB 2: TRADE
    with tab2:
        st.subheader("Quick Trade")

        col1, col2 = st.columns(2)

        # BUY SECTION
        with col1:
            st.markdown("### Buy")

            with st.form(f"quick_buy_{ticker}"):
                quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)
                use_limit = st.checkbox("Limit Order")

                if use_limit:
                    limit_price = st.number_input("Limit Price", min_value=0.01, value=current_price, step=0.01)
                else:
                    limit_price = None
                    st.info(f"Market Price: ${current_price:.2f}")

                total = quantity * (limit_price if limit_price else current_price)
                st.markdown(f"**Total: ${total:.2f}**")

                submit = st.form_submit_button("Buy Now", use_container_width=True)

                if submit:
                    if use_limit:
                        # Create pending order
                        order_id = service.db.add_pending_order(ticker, 'BUY', quantity, limit_price)
                        if order_id:
                            st.success(f"Limit buy order placed! Order ID: {order_id}")
                        else:
                            st.error("Failed to place order")
                    else:
                        # Execute market order
                        success, msg = service.buy_stock(ticker, quantity, current_price)
                        if success:
                            st.success(msg)
                            st.balloons()
                        else:
                            st.error(msg)

        # SELL SECTION
        with col2:
            st.markdown("### Sell")

            # Check current position
            positions = service.get_positions()
            position = next((p for p in positions if p.ticker == ticker), None)

            if position:
                st.info(f"You own {position.quantity:.2f} shares")

                with st.form(f"quick_sell_{ticker}"):
                    sell_qty = st.number_input("Quantity", min_value=0.01,
                                              max_value=float(position.quantity),
                                              value=min(1.0, float(position.quantity)),
                                              step=0.01)
                    use_limit_sell = st.checkbox("Limit Order", key="sell_limit")

                    if use_limit_sell:
                        limit_price_sell = st.number_input("Limit Price", min_value=0.01,
                                                          value=current_price, step=0.01,
                                                          key="sell_limit_price")
                    else:
                        limit_price_sell = None
                        st.info(f"Market Price: ${current_price:.2f}")

                    total_sell = sell_qty * (limit_price_sell if limit_price_sell else current_price)
                    st.markdown(f"**Total: ${total_sell:.2f}**")

                    submit_sell = st.form_submit_button("Sell Now", use_container_width=True)

                    if submit_sell:
                        if use_limit_sell:
                            order_id = service.db.add_pending_order(ticker, 'SELL', sell_qty, limit_price_sell)
                            if order_id:
                                st.success(f"Limit sell order placed! Order ID: {order_id}")
                            else:
                                st.error("Failed to place order")
                        else:
                            success, msg = service.sell_stock(ticker, sell_qty, current_price)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
            else:
                st.warning("You don't own any shares of this stock")

    # TAB 3: ORDERS
    with tab3:
        # Order execution controls
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Pending Orders")

        with col2:
            # Check and execute orders button
            if st.button("Check & Execute Orders", use_container_width=True, type="primary"):
                execution_service = OrderExecutionService(service)
                executed = execution_service.check_and_execute_orders()

                if executed:
                    st.success(f"Executed {len(executed)} order(s)!")
                    for order_id, msg, success in executed:
                        if success:
                            st.info(msg)
                        else:
                            st.warning(msg)
                    st.rerun()
                else:
                    st.info("No orders ready to execute")

        st.divider()

        orders = service.db.get_pending_orders_by_ticker(ticker)

        if orders:
            for order in orders:
                # Get order status
                order_status = get_order_status_message(order, current_price)

                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    order_type_color = "[BUY]" if order.order_type == 'BUY' else "[SELL]"
                    limit_display = f"@ ${order.limit_price:.2f}" if order.limit_price else "MARKET"
                    st.markdown(f"""
                    <div class="order-card">
                        {order_type_color} <b>{order.order_type}</b> {order.quantity:.2f} shares {limit_display}<br/>
                        <small>Order #{order.id} - {order.created_at}</small><br/>
                        <small>{order_status}</small>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if st.button("Cancel", key=f"cancel_{order.id}"):
                        if service.db.cancel_order(order.id):
                            st.success("Order cancelled")
                            st.rerun()

                with col3:
                    if st.button("Delete", key=f"delete_{order.id}"):
                        if service.db.delete_order(order.id):
                            st.success("Order deleted")
                            st.rerun()
        else:
            st.info("No pending orders for this stock")

        # Show recent transactions
        st.divider()
        st.subheader("Recent Transactions")

        transactions = service.get_transactions(ticker=ticker)

        if transactions[:5]:  # Show last 5
            import pandas as pd

            df = pd.DataFrame([{
                'Date': t.transaction_date,
                'Type': t.transaction_type,
                'Quantity': f"{t.quantity:.2f}",
                'Price': f"${t.price:.2f}",
                'Total': f"${t.quantity * t.price:.2f}"
            } for t in transactions[:5]])

            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No transaction history")


def render_portfolio_view():
    """Render portfolio overview with AI assistant"""
    st.markdown('<div class="main-header">My Portfolio</div>', unsafe_allow_html=True)

    service = st.session_state.service
    summary = service.get_portfolio_summary()

    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Value", f"${summary['total_market_value']:,.2f}")
    with col2:
        st.metric("Invested", f"${summary['total_cost_basis']:,.2f}")
    with col3:
        delta_color = "normal" if summary['total_unrealized_pnl'] >= 0 else "inverse"
        st.metric("P&L", f"${summary['total_unrealized_pnl']:,.2f}",
                 f"{summary['total_unrealized_pnl_pct']:.2f}%", delta_color=delta_color)
    with col4:
        st.metric("Positions", summary['num_positions'])

    st.divider()

    # Tabs for holdings and AI assistant
    tab1, tab2 = st.tabs(["Holdings", "AI Assistant"])

    with tab1:
        # Holdings table
        if summary['positions']:
            st.subheader("Your Positions")

            for position in summary['positions']:
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                pnl_symbol = "+" if position.unrealized_pnl >= 0 else "-"
                pnl_color = "normal" if position.unrealized_pnl >= 0 else "inverse"

                with col1:
                    st.markdown(f"**{position.ticker}**")
                    st.caption(f"{position.quantity:.2f} shares @ ${position.avg_cost:.2f}")

                with col2:
                    st.metric("Current", f"${position.current_price:.2f}")

                with col3:
                    st.metric("Value", f"${position.market_value:.2f}")

                with col4:
                    st.metric("P&L", f"${abs(position.unrealized_pnl):.2f}",
                             f"{position.unrealized_pnl_pct:+.2f}%", delta_color=pnl_color)

                if st.button("View Details", key=f"view_detail_{position.ticker}"):
                    st.session_state.selected_stock = position.ticker
                    st.session_state.view_mode = "stock_detail"
                    st.rerun()

                st.divider()
        else:
            st.info("Your portfolio is empty. Start trading to build your portfolio!")

    with tab2:
        # AI Portfolio Assistant
        st.subheader("Portfolio Insight Agent")
        st.caption("Ask questions about your portfolio risk, diversification, and performance")

        # Initialize agent if not exists
        if st.session_state.agent is None:
            try:
                st.session_state.agent = PortfolioInsightAgent(service=service)
            except Exception as e:
                st.error(f"Could not initialize AI agent: {str(e)}")
                st.info("Make sure you have ANTHROPIC_API_KEY or OPENAI_API_KEY set in your .env file")
                return

        # Quick analysis button
        if summary['positions']:
            if st.button("Analyze My Portfolio", use_container_width=True, type="primary"):
                with st.spinner("Analyzing your portfolio..."):
                    try:
                        analysis, metrics, positions = st.session_state.agent.analyze_live_portfolio()

                        # Display text analysis
                        st.markdown(analysis)

                        # Display visualizations
                        if metrics and positions:
                            st.divider()
                            st.subheader("Portfolio Visualizations")

                            col1, col2 = st.columns(2)

                            with col1:
                                # Holdings pie chart
                                holdings_data = {p.ticker: p.market_value for p in positions}
                                fig_holdings = px.pie(
                                    values=list(holdings_data.values()),
                                    names=list(holdings_data.keys()),
                                    title="Portfolio Allocation by Holdings"
                                )
                                st.plotly_chart(fig_holdings, use_container_width=True)

                            with col2:
                                # Sector pie chart
                                if metrics.sector_concentration:
                                    fig_sectors = px.pie(
                                        values=list(metrics.sector_concentration.values()),
                                        names=list(metrics.sector_concentration.keys()),
                                        title="Portfolio Allocation by Sector"
                                    )
                                    st.plotly_chart(fig_sectors, use_container_width=True)

                            # Risk metrics bar chart
                            st.subheader("Risk Metrics Comparison")
                            risk_metrics_data = {
                                "Volatility": metrics.volatility,
                                "Beta": metrics.beta * 10,  # Scale for visibility
                                "Max Drawdown": abs(metrics.max_drawdown),
                                "Diversification": metrics.diversification_score
                            }

                            fig_risk = px.bar(
                                x=list(risk_metrics_data.keys()),
                                y=list(risk_metrics_data.values()),
                                title="Risk Profile (scaled for comparison)",
                                labels={'x': 'Metric', 'y': 'Value'}
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)

                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")

        st.divider()

        # Chat interface
        st.markdown("**Ask a Question**")

        # Example queries
        with st.expander("Example Questions"):
            st.markdown("""
            - "What's the overall risk level of my portfolio?"
            - "How diversified am I?"
            - "What are the main strengths and weaknesses?"
            - "Should I rebalance my portfolio?"
            - "Which holdings have the most risk?"
            """)

        # Chat input
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What's my portfolio's risk level?",
            key="portfolio_question"
        )

        if st.button("Ask", use_container_width=True):
            if user_question:
                with st.spinner("Thinking..."):
                    try:
                        # Get portfolio context
                        positions_str = ", ".join([
                            f"{p.ticker} ({p.market_value/summary['total_market_value']*100:.1f}%)"
                            for p in summary['positions']
                        ])

                        # Create context-aware query
                        context_query = f"""My current portfolio: {positions_str}

Question: {user_question}"""

                        response, st.session_state.chat_history = st.session_state.agent.chat(
                            context_query,
                            st.session_state.chat_history
                        )

                        st.markdown(f"**Agent:** {response}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question")

        # Show chat history
        if st.session_state.chat_history:
            with st.expander("Chat History", expanded=False):
                for idx, msg in enumerate(st.session_state.chat_history):
                    if msg.type == "human":
                        st.markdown(f"**You:** {msg.content}")
                    else:
                        st.markdown(f"**Agent:** {msg.content}")
                    if idx < len(st.session_state.chat_history) - 1:
                        st.divider()


def main():
    """Main application"""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Trading Platform")

        # Navigation
        # Only change view_mode if user clicks navigation, not on every rerun
        current_nav = st.session_state.get("nav", "market")

        view = st.radio(
            "Navigate",
            options=["market", "portfolio", "settings"],
            format_func=lambda x: {"market": "Market", "portfolio": "Portfolio", "settings": "Settings"}[x],
            key="nav",
            index=["market", "portfolio", "settings"].index(current_nav) if current_nav in ["market", "portfolio", "settings"] else 0
        )

        # Only update view_mode if navigation actually changed and we're not viewing stock details
        if view != current_nav:
            if view == "market":
                st.session_state.view_mode = "market"
                st.session_state.selected_stock = None
            elif view == "portfolio":
                st.session_state.view_mode = "portfolio"
                st.session_state.selected_stock = None
            st.rerun()

        st.divider()

        # Quick stats
        service = st.session_state.service
        summary = service.get_portfolio_summary()

        st.metric("Portfolio Value", f"${summary['total_market_value']:,.2f}")
        st.metric("Day P&L", f"${summary['total_unrealized_pnl']:,.2f}")

        # Pending orders count
        all_orders = service.db.get_all_pending_orders()
        if all_orders:
            st.info(f"{len(all_orders)} pending orders")

    # Main content routing
    if st.session_state.view_mode == "stock_detail" and st.session_state.selected_stock:
        render_stock_detail()
    elif st.session_state.view_mode == "portfolio" or view == "portfolio":
        render_portfolio_view()
    elif st.session_state.nav == "settings" or view == "settings":
        render_settings()
    elif st.session_state.view_mode == "market" or view == "market":
        render_market_view()
    else:
        # Default fallback
        render_market_view()


def render_settings():
    """Render settings page"""
    st.markdown('<div class="main-header">Settings</div>', unsafe_allow_html=True)

    service = st.session_state.service

    tab1, tab2 = st.tabs(["Watchlist", "All Orders"])

    with tab1:
        st.subheader("Manage Watchlist")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Add Stock")
            with st.form("add_stock_watchlist"):
                ticker = st.text_input("Ticker Symbol").upper()
                submit = st.form_submit_button("Add to Watchlist")

                if submit and ticker:
                    success, msg = service.add_stock(ticker)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with col2:
            st.markdown("### Remove Stock")
            stocks = service.list_stocks()
            if stocks:
                stock_options = {s.ticker: f"{s.ticker} - {s.company_name}" for s in stocks}

                with st.form("remove_stock_watchlist"):
                    del_ticker = st.selectbox("Select Stock", options=list(stock_options.keys()),
                                             format_func=lambda x: stock_options[x])
                    submit = st.form_submit_button("Remove")

                    if submit:
                        success, msg = service.delete_stock(del_ticker)
                        if success:
                            st.warning(msg)
                            st.rerun()

        # Display watchlist
        st.divider()
        st.subheader("Current Watchlist")

        if stocks:
            for stock in stocks:
                st.write(f"**{stock.ticker}** - {stock.company_name} ({stock.sector or 'N/A'})")
        else:
            st.info("Watchlist is empty")

    with tab2:
        st.subheader("All Pending Orders")

        orders = service.db.get_all_pending_orders()

        if orders:
            import pandas as pd

            df = pd.DataFrame([{
                'ID': o.id,
                'Ticker': o.ticker,
                'Type': o.order_type,
                'Quantity': f"{o.quantity:.2f}",
                'Limit Price': f"${o.limit_price:.2f}" if o.limit_price else "Market",
                'Created': o.created_at
            } for o in orders])

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Bulk actions
            st.divider()
            if st.button("Cancel All Orders"):
                for order in orders:
                    service.db.cancel_order(order.id)
                st.success("All orders cancelled")
                st.rerun()
        else:
            st.info("No pending orders")


if __name__ == "__main__":
    main()
