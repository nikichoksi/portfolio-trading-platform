"""
Streamlit UI for Portfolio Insight Agent.
Interactive conversational interface for portfolio analysis.
"""

import streamlit as st
import os
from typing import List
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from agents.portfolio_agent import PortfolioInsightAgent
from core.portfolio_metrics import PortfolioAnalyzer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Portfolio Insight Agent",
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
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
    }
    .agent-message {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None


def check_api_keys() -> tuple[bool, str]:
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


def create_metrics_visualization(metrics):
    """Create visualizations for portfolio metrics"""
    col1, col2 = st.columns(2)

    with col1:
        # Risk-Return Scatter
        fig_risk_return = go.Figure()
        fig_risk_return.add_trace(go.Scatter(
            x=[metrics.annual_volatility],
            y=[metrics.annual_return],
            mode='markers',
            marker=dict(size=20, color='#3B82F6'),
            name='Portfolio',
            text=[f'Sharpe: {metrics.sharpe_ratio:.2f}'],
            hovertemplate='<b>Your Portfolio</b><br>' +
                         'Return: %{y:.2%}<br>' +
                         'Volatility: %{x:.2%}<br>' +
                         '%{text}<extra></extra>'
        ))

        fig_risk_return.update_layout(
            title='Risk-Return Profile',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Return',
            hovermode='closest',
            height=300
        )

        st.plotly_chart(fig_risk_return, use_container_width=True)

    with col2:
        # Holdings Pie Chart
        fig_holdings = go.Figure(data=[go.Pie(
            labels=list(metrics.holdings.keys()),
            values=list(metrics.holdings.values()),
            hole=0.4
        )])

        fig_holdings.update_layout(
            title='Portfolio Allocation',
            height=300
        )

        st.plotly_chart(fig_holdings, use_container_width=True)

    # Risk Metrics Gauge
    st.subheader("Risk Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics.sharpe_ratio:.3f}",
            delta="Good" if metrics.sharpe_ratio > 1 else "Fair" if metrics.sharpe_ratio > 0 else "Poor"
        )

    with col2:
        st.metric(
            label="Beta",
            value=f"{metrics.beta:.3f}",
            delta="Defensive" if metrics.beta < 0.8 else "Aggressive" if metrics.beta > 1.2 else "Market-like"
        )

    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics.max_drawdown:.2%}",
            delta="High Risk" if abs(metrics.max_drawdown) > 0.3 else "Moderate"
        )

    with col4:
        st.metric(
            label="VaR (95%)",
            value=f"{metrics.var_95:.2%}",
            delta=None
        )


def render_chat_history():
    """Render chat history"""
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>
                {message.content}
            </div>
            """, unsafe_allow_html=True)
        else:  # Agent message
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>Agent:</strong><br>
                {message.content}
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application"""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📊 Portfolio Insight Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Conversational Portfolio Risk Analysis</div>',
        unsafe_allow_html=True
    )

    # Check API keys
    has_key, provider = check_api_keys()

    if not has_key:
        st.error("""
        ⚠️ **API Key Required**

        Please set up your API key in the `.env` file:

        1. Copy `.env.example` to `.env`
        2. Add your API key:
           - `ANTHROPIC_API_KEY=your_key_here` (for Claude)
           - OR `OPENAI_API_KEY=your_key_here` (for GPT)
        3. Restart the application

        Get your API key:
        - Anthropic: https://console.anthropic.com/
        - OpenAI: https://platform.openai.com/
        """)
        return

    # Initialize agent
    if st.session_state.agent is None:
        with st.spinner("Initializing Portfolio Insight Agent..."):
            if not initialize_agent(provider):
                return

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This agent analyzes your investment portfolio and provides:
        - **Risk metrics** (volatility, beta, VaR)
        - **Return analysis** (Sharpe ratio, drawdown)
        - **AI insights** on strengths and weaknesses
        - **Conversational Q&A** about your investments
        """)

        st.header("Example Queries")
        st.code("""
• Analyze my portfolio: 40% AAPL, 30% MSFT, 30% GOOGL

• What are the main risks?

• How diversified is this?

• Compare with S&P 500

• Should I rebalance?
        """)

        st.header("Quick Analysis")
        with st.form("quick_analysis"):
            example = st.selectbox(
                "Try an example portfolio:",
                [
                    "40% AAPL, 30% MSFT, 30% GOOGL",
                    "50% SPY, 30% QQQ, 20% TLT",
                    "60% VTI, 30% VXUS, 10% BND",
                    "25% AAPL, 25% TSLA, 25% NVDA, 25% AMZN"
                ]
            )
            if st.form_submit_button("Analyze"):
                st.session_state.quick_query = f"Analyze my portfolio: {example}"

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_metrics = None
            st.rerun()

    # Main content
    tab1, tab2 = st.tabs(["💬 Chat", "📈 Detailed Metrics"])

    with tab1:
        st.subheader("Chat with Your Portfolio Analyst")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            render_chat_history()

        # Chat input
        user_input = st.chat_input("Ask about your portfolio or describe it for analysis...")

        # Handle quick analysis from sidebar
        if hasattr(st.session_state, 'quick_query'):
            user_input = st.session_state.quick_query
            delattr(st.session_state, 'quick_query')

        if user_input:
            # Add user message to history
            with st.spinner("Analyzing..."):
                try:
                    response, updated_history = st.session_state.agent.chat(
                        user_input,
                        st.session_state.chat_history
                    )
                    st.session_state.chat_history = updated_history
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with tab2:
        st.subheader("Detailed Portfolio Metrics")

        if st.session_state.last_metrics:
            create_metrics_visualization(st.session_state.last_metrics)
        else:
            st.info("👈 Start by analyzing a portfolio in the Chat tab!")

            st.markdown("""
            ### How to use:

            1. **Describe your portfolio** in the chat (e.g., "40% AAPL, 30% MSFT, 30% GOOGL")
            2. **Ask questions** about risk, diversification, or performance
            3. **View detailed metrics** and visualizations here

            The agent will calculate:
            - Annual return and volatility
            - Sharpe ratio (risk-adjusted return)
            - Beta (market sensitivity)
            - Maximum drawdown
            - Value at Risk (VaR)
            - And more!
            """)

    # Footer
    st.divider()
    st.caption(f"Powered by {'Claude' if provider == 'anthropic' else 'GPT'} via LangChain | "
               f"Financial data from Yahoo Finance")


if __name__ == "__main__":
    main()
