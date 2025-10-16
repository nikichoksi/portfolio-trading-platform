"""
Enhanced Streamlit Dashboard for Portfolio Management.
Includes CRUD operations, live positions, buy/sell interface, and comprehensive visualizations.
"""

import streamlit as st
import os
from typing import List
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agents.portfolio_agent import PortfolioInsightAgent
from core.portfolio_metrics import PortfolioAnalyzer
from services.portfolio_service import PortfolioService
from utils.visualizations import (
    create_performance_chart, create_drawdown_chart, create_correlation_heatmap,
    create_sector_pie_chart, create_risk_return_scatter, create_rolling_metrics_chart,
    create_var_chart, create_positions_table_chart, create_portfolio_allocation_pie,
    create_pnl_bar_chart, create_pnl_percentage_bar, create_portfolio_value_gauge,
    create_portfolio_timeline
)
from utils.sector_analysis import SectorAnalyzer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Portfolio Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    .success-box {
        padding: 1rem;
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 1rem;
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "portfolio_service" not in st.session_state:
        st.session_state.portfolio_service = PortfolioService()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "refresh_trigger" not in st.session_state:
        st.session_state.refresh_trigger = 0


def check_api_keys():
    """Check if API keys are configured"""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        return True, "anthropic"
    elif openai_key:
        return True, "openai"
    else:
        return False, ""


def initialize_agent(provider: str):
    """Initialize the portfolio agent"""
    try:
        use_openai = provider == "openai"
        st.session_state.agent = PortfolioInsightAgent(use_openai=use_openai)
        return True
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return False


def render_chat_history():
    """Render chat history"""
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        else:  # Agent message
            with st.chat_message("assistant", avatar="🤖"):
                content = message.content
                if isinstance(content, list):
                    if len(content) > 0 and isinstance(content[0], dict) and 'text' in content[0]:
                        content = content[0]['text']
                    else:
                        content = str(content)
                st.markdown(content)


def main():
    """Main application"""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📊 Portfolio Management Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Complete Portfolio Management with AI Insights</div>',
        unsafe_allow_html=True
    )

    service = st.session_state.portfolio_service

    # Sidebar
    with st.sidebar:
        st.header("Portfolio Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.refresh_trigger += 1
            st.rerun()

        st.divider()

        # Quick stats
        st.subheader("Quick Stats")
        summary = service.get_portfolio_summary()

        st.metric(
            "Total Value",
            f"${summary['total_market_value']:,.2f}",
            f"{summary['total_unrealized_pnl_pct']:.2f}%"
        )
        st.metric("Positions", summary['num_positions'])
        st.metric(
            "Total P&L",
            f"${summary['total_unrealized_pnl']:,.2f}",
            None if summary['total_unrealized_pnl'] >= 0 else "Loss"
        )

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Dashboard",
        "💼 Holdings",
        "🔄 Buy/Sell",
        "📋 Transactions",
        "🤖 AI Chat",
        "⚙️ Manage Stocks"
    ])

    # TAB 1: DASHBOARD
    with tab1:
        st.subheader("Portfolio Overview")

        positions = service.get_positions()

        if not positions:
            st.info("📝 Your portfolio is empty. Add stocks and transactions to get started!")
        else:
            # Performance metrics
            metrics = service.get_performance_metrics()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Invested", f"${metrics['total_invested']:,.2f}")
            with col2:
                st.metric("Current Value", f"${metrics['current_value']:,.2f}")
            with col3:
                delta_color = "normal" if metrics['total_return'] >= 0 else "inverse"
                st.metric("Total Return", f"${metrics['total_return']:,.2f}",
                         f"{metrics['total_return_pct']:.2f}%", delta_color=delta_color)
            with col4:
                st.metric("Positions", metrics['num_positions'])

            st.divider()

            # Charts row 1
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_portfolio_value_gauge(
                        metrics['current_value'],
                        metrics['total_invested']
                    ),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_portfolio_allocation_pie(positions),
                    use_container_width=True
                )

            # Charts row 2
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_pnl_bar_chart(positions),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_pnl_percentage_bar(positions),
                    use_container_width=True
                )

            # Top gainers and losers
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Top Gainer")
                if metrics['top_gainer']:
                    gainer = metrics['top_gainer']
                    st.success(f"**{gainer.ticker}**: +${gainer.unrealized_pnl:.2f} ({gainer.unrealized_pnl_pct:.2f}%)")
                else:
                    st.info("No data")

            with col2:
                st.subheader("📉 Top Loser")
                if metrics['top_loser']:
                    loser = metrics['top_loser']
                    st.error(f"**{loser.ticker}**: ${loser.unrealized_pnl:.2f} ({loser.unrealized_pnl_pct:.2f}%)")
                else:
                    st.info("No data")

    # TAB 2: HOLDINGS
    with tab2:
        st.subheader("Current Holdings")

        positions = service.get_positions()

        if not positions:
            st.info("No holdings found. Make some purchases to see your positions here.")
        else:
            # Positions table
            st.plotly_chart(
                create_positions_table_chart(positions),
                use_container_width=True
            )

            # Sector allocation
            st.divider()
            sector_allocation = service.get_sector_allocation()

            if sector_allocation:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.plotly_chart(
                        create_sector_pie_chart(sector_allocation),
                        use_container_width=True
                    )

                with col2:
                    st.subheader("Sector Breakdown")
                    for sector, pct in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{sector}**: {pct:.2f}%")

    # TAB 3: BUY/SELL
    with tab3:
        st.subheader("Execute Trades")

        col1, col2 = st.columns(2)

        # BUY FORM
        with col1:
            st.markdown("### 📈 Buy Stock")
            with st.form("buy_form"):
                buy_ticker = st.text_input("Ticker Symbol", key="buy_ticker").upper()
                buy_quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01, key="buy_quantity")
                buy_price = st.number_input("Price per Share", min_value=0.01, value=100.0, step=0.01, key="buy_price")
                buy_date = st.date_input("Transaction Date", value=datetime.now(), key="buy_date")
                buy_notes = st.text_area("Notes (optional)", key="buy_notes")

                submitted = st.form_submit_button("🛒 Execute Buy", use_container_width=True)

                if submitted and buy_ticker:
                    success, message = service.buy_stock(
                        ticker=buy_ticker,
                        quantity=buy_quantity,
                        price=buy_price,
                        date=buy_date.strftime("%Y-%m-%d"),
                        notes=buy_notes if buy_notes else None
                    )

                    if success:
                        st.success(message)
                        st.session_state.refresh_trigger += 1
                        st.rerun()
                    else:
                        st.error(message)

        # SELL FORM
        with col2:
            st.markdown("### 📉 Sell Stock")

            positions = service.get_positions()
            position_options = {p.ticker: f"{p.ticker} ({p.quantity:.2f} shares available)"
                               for p in positions}

            with st.form("sell_form"):
                if position_options:
                    sell_ticker = st.selectbox("Select Position", options=list(position_options.keys()),
                                              format_func=lambda x: position_options[x], key="sell_ticker")
                else:
                    st.warning("No positions available to sell")
                    sell_ticker = None

                sell_quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01, key="sell_quantity")
                sell_price = st.number_input("Price per Share", min_value=0.01, value=100.0, step=0.01, key="sell_price")
                sell_date = st.date_input("Transaction Date", value=datetime.now(), key="sell_date")
                sell_notes = st.text_area("Notes (optional)", key="sell_notes")

                submitted = st.form_submit_button("💰 Execute Sell", use_container_width=True,
                                                 disabled=not position_options)

                if submitted and sell_ticker:
                    success, message = service.sell_stock(
                        ticker=sell_ticker,
                        quantity=sell_quantity,
                        price=sell_price,
                        date=sell_date.strftime("%Y-%m-%d"),
                        notes=sell_notes if sell_notes else None
                    )

                    if success:
                        st.success(message)
                        st.session_state.refresh_trigger += 1
                        st.rerun()
                    else:
                        st.error(message)

        # Live prices section
        st.divider()
        st.subheader("💹 Live Stock Prices")

        col1, col2 = st.columns([3, 1])

        with col1:
            price_ticker = st.text_input("Enter ticker to get live price", key="price_check").upper()

        with col2:
            st.write("")
            st.write("")
            check_price = st.button("Get Price", use_container_width=True)

        if check_price and price_ticker:
            with st.spinner(f"Fetching price for {price_ticker}..."):
                info = service.get_stock_info(price_ticker)

                if info and info.get('current_price', 0) > 0:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Current Price", f"${info['current_price']:.2f}")
                    with col2:
                        change = info['current_price'] - info['previous_close']
                        change_pct = (change / info['previous_close'] * 100) if info['previous_close'] > 0 else 0
                        st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
                    with col3:
                        st.metric("Day High", f"${info['day_high']:.2f}")
                    with col4:
                        st.metric("Day Low", f"${info['day_low']:.2f}")

                    st.info(f"**{info['company_name']}** | Sector: {info['sector']} | Industry: {info['industry']}")
                else:
                    st.error("Unable to fetch price information")

    # TAB 4: TRANSACTIONS
    with tab4:
        st.subheader("Transaction History")

        transactions = service.get_transactions(limit=100)

        if not transactions:
            st.info("No transactions found.")
        else:
            # Display transactions in a table
            import pandas as pd

            df = pd.DataFrame([{
                'ID': t.id,
                'Date': t.transaction_date,
                'Ticker': t.ticker,
                'Type': t.transaction_type,
                'Quantity': f"{t.quantity:.2f}",
                'Price': f"${t.price:.2f}",
                'Total': f"${t.quantity * t.price:.2f}",
                'Notes': t.notes or '-'
            } for t in transactions])

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Delete transaction
            st.divider()
            col1, col2 = st.columns([3, 1])

            with col1:
                transaction_id = st.number_input("Transaction ID to delete", min_value=1, step=1, key="del_txn_id")

            with col2:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete", use_container_width=True):
                    success, message = service.delete_transaction(int(transaction_id))
                    if success:
                        st.success(message)
                        st.session_state.refresh_trigger += 1
                        st.rerun()
                    else:
                        st.error(message)

    # TAB 5: AI CHAT
    with tab5:
        st.subheader("Chat with AI Portfolio Analyst")

        # Check API keys
        has_key, provider = check_api_keys()

        if not has_key:
            st.error("⚠️ API Key Required for AI Chat. Please configure ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")
        else:
            # Initialize agent
            if st.session_state.agent is None:
                with st.spinner("Initializing AI agent..."):
                    if not initialize_agent(provider):
                        st.stop()

            # Display chat history
            chat_container = st.container()
            with chat_container:
                render_chat_history()

            # Chat input
            user_input = st.chat_input("Ask about your portfolio...")

            if user_input:
                with st.spinner("Analyzing..."):
                    try:
                        # Get current portfolio for context
                        positions = service.get_positions()
                        if positions:
                            portfolio_str = ", ".join([f"{p.ticker}" for p in positions])
                            context = f"Current portfolio: {portfolio_str}. "
                            full_input = context + user_input
                        else:
                            full_input = user_input

                        response, updated_history = st.session_state.agent.chat(
                            full_input,
                            st.session_state.chat_history
                        )

                        st.session_state.chat_history = updated_history
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # TAB 6: MANAGE STOCKS
    with tab6:
        st.subheader("Manage Stock Database")

        col1, col2 = st.columns(2)

        # ADD STOCK
        with col1:
            st.markdown("### ➕ Add New Stock")

            with st.form("add_stock_form"):
                ticker = st.text_input("Ticker Symbol").upper()
                company_name = st.text_input("Company Name (optional)")
                sector = st.text_input("Sector (optional)")

                submitted = st.form_submit_button("Add Stock", use_container_width=True)

                if submitted and ticker:
                    success, message = service.add_stock(
                        ticker=ticker,
                        company_name=company_name if company_name else None,
                        sector=sector if sector else None
                    )

                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        # DELETE STOCK
        with col2:
            st.markdown("### 🗑️ Delete Stock")

            stocks = service.list_stocks()
            stock_options = {s.ticker: f"{s.ticker} - {s.company_name}" for s in stocks}

            with st.form("delete_stock_form"):
                if stock_options:
                    del_ticker = st.selectbox("Select Stock", options=list(stock_options.keys()),
                                             format_func=lambda x: stock_options[x])
                else:
                    st.warning("No stocks in database")
                    del_ticker = None

                submitted = st.form_submit_button("Delete Stock", use_container_width=True,
                                                 disabled=not stock_options)

                if submitted and del_ticker:
                    success, message = service.delete_stock(del_ticker)

                    if success:
                        st.warning(message)
                        st.rerun()
                    else:
                        st.error(message)

        # LIST ALL STOCKS
        st.divider()
        st.markdown("### 📚 Stock Database")

        stocks = service.list_stocks()

        if stocks:
            import pandas as pd

            df = pd.DataFrame([{
                'Ticker': s.ticker,
                'Company': s.company_name,
                'Sector': s.sector or 'N/A'
            } for s in stocks])

            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in database")

    # Footer
    st.divider()
    st.caption("Portfolio Management Dashboard | Powered by AI | Real-time Data from Yahoo Finance")


if __name__ == "__main__":
    main()
