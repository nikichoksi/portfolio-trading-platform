"""
Portfolio Insight Agent using LangChain.
Provides conversational portfolio analysis with AI-generated insights.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.portfolio_analytics import PortfolioAnalyzer, PortfolioMetrics, generate_portfolio_narrative, identify_portfolio_strengths_weaknesses
    LEGACY_MODE = False
except ImportError:
    from core.portfolio_metrics import PortfolioAnalyzer as CoreAnalyzer, PortfolioMetrics as CoreMetrics
    from utils.sector_analysis import SectorAnalyzer
    PortfolioMetrics = CoreMetrics  # Alias for consistency
    LEGACY_MODE = True

# Load environment variables
load_dotenv()


class PortfolioInsightAgent:
    """
    Conversational agent for portfolio analysis.
    Uses LangChain with tool-calling capabilities.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.1,
        use_openai: bool = False,
        service=None  # Portfolio service for direct integration
    ):
        """
        Initialize the Portfolio Insight Agent.

        Args:
            model: Model name (Claude or GPT)
            temperature: Model temperature (0-1)
            use_openai: If True, use OpenAI instead of Anthropic
            service: Optional PortfolioService for direct integration
        """
        if LEGACY_MODE:
            self.analyzer = CoreAnalyzer()
            self.sector_analyzer = SectorAnalyzer()
        else:
            self.analyzer = PortfolioAnalyzer()

        self.service = service
        self.use_openai = use_openai

        # Initialize LLM
        if use_openai:
            self.llm = ChatOpenAI(
                model=model if "gpt" in model else "gpt-4-turbo-preview",
                temperature=temperature
            )
        else:
            self.llm = ChatAnthropic(
                model=model,
                temperature=temperature
            )

        # Create tools
        self.tools = self._create_tools()

        # Create agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent to use"""

        def analyze_portfolio_tool(query: str) -> str:
            """
            Analyze a portfolio from a natural language description.
            Example: "40% AAPL, 30% MSFT, 30% GOOGL"
            """
            try:
                holdings = self.analyzer.parse_portfolio(query)
                if not holdings:
                    return "Could not parse portfolio. Please provide format like: 40% AAPL, 30% MSFT, 30% GOOGL"

                metrics = self.analyzer.analyze_portfolio(holdings)
                return self._format_metrics(metrics)

            except Exception as e:
                return f"Error analyzing portfolio: {str(e)}"

        def get_correlation_matrix(tickers: str) -> str:
            """
            Get correlation matrix for given tickers.
            Example: "AAPL,MSFT,GOOGL"
            """
            try:
                ticker_list = [t.strip().upper() for t in tickers.split(",")]
                corr_matrix = self.analyzer.get_asset_correlations(ticker_list)
                return corr_matrix.to_string()
            except Exception as e:
                return f"Error getting correlations: {str(e)}"

        def analyze_sector_exposure(portfolio_str: str) -> str:
            """
            Analyze sector exposure for a portfolio.
            Example: "40% AAPL, 30% MSFT, 30% GOOGL"
            """
            try:
                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                sector_summary = self.sector_analyzer.get_sector_summary(holdings)
                return self.sector_analyzer.format_sector_analysis(sector_summary)

            except Exception as e:
                return f"Error analyzing sectors: {str(e)}"

        tools = [
            Tool(
                name="analyze_portfolio",
                func=analyze_portfolio_tool,
                description=(
                    "Analyzes a portfolio and returns comprehensive risk-return metrics. "
                    "Input should be a portfolio description like '40% AAPL, 30% MSFT, 30% GOOGL'. "
                    "Returns annual return, volatility, Sharpe ratio, max drawdown, beta, VaR, and CVaR."
                )
            ),
            Tool(
                name="get_correlations",
                func=get_correlation_matrix,
                description=(
                    "Returns correlation matrix for given stock tickers. "
                    "Input should be comma-separated tickers like 'AAPL,MSFT,GOOGL'. "
                    "Useful for understanding diversification between assets."
                )
            ),
            Tool(
                name="analyze_sectors",
                func=analyze_sector_exposure,
                description=(
                    "Analyzes portfolio sector exposure and concentration. "
                    "Input should be a portfolio description like '40% AAPL, 30% MSFT, 30% GOOGL'. "
                    "Returns sector breakdown, concentration metrics, and risk profile."
                )
            )
        ]

        return tools

    def _format_metrics(self, metrics: PortfolioMetrics) -> str:
        """Format portfolio metrics into a readable string"""
        # Handle both legacy and new metrics formats
        if LEGACY_MODE:
            holdings_str = ", ".join([f"{k}: {v:.1%}" for k, v in metrics.holdings.items()])

            # Format advanced metrics if available
            advanced_metrics = ""
            if metrics.sortino_ratio is not None:
                advanced_metrics = f"""
Advanced Risk-Adjusted Metrics:
- Sortino Ratio: {metrics.sortino_ratio:.3f}
- Calmar Ratio: {metrics.calmar_ratio:.3f}
- Information Ratio: {metrics.information_ratio:.3f}
- Downside Deviation: {metrics.downside_deviation:.2%}
"""

            return f"""
Portfolio Analysis Results:
==========================

Holdings: {holdings_str}
Analysis Period: {metrics.period_days} days

Risk-Return Metrics:
- Annual Return: {metrics.annual_return:.2%}
- Annual Volatility: {metrics.annual_volatility:.2%}
- Sharpe Ratio: {metrics.sharpe_ratio:.3f}
- Beta: {metrics.beta:.3f}

Risk Metrics:
- Maximum Drawdown: {metrics.max_drawdown:.2%}
- Value at Risk (95%): {metrics.var_95:.2%}
- Conditional VaR (95%): {metrics.cvar_95:.2%}
{advanced_metrics}
Interpretation:
- Sharpe Ratio: {'Excellent (>2)' if metrics.sharpe_ratio > 2 else 'Good (1-2)' if metrics.sharpe_ratio > 1 else 'Fair (0-1)' if metrics.sharpe_ratio > 0 else 'Poor (<0)'}
- Volatility: {'High (>25%)' if metrics.annual_volatility > 0.25 else 'Moderate (15-25%)' if metrics.annual_volatility > 0.15 else 'Low (<15%)'}
- Beta: {'Aggressive (>1.2)' if metrics.beta > 1.2 else 'Market-like (0.8-1.2)' if metrics.beta > 0.8 else 'Defensive (<0.8)'}
"""
        else:
            # New analytics format
            top_holdings_str = ", ".join([f"{ticker}: {weight:.1%}" for ticker, weight in metrics.top_holdings[:3]])

            return f"""
Portfolio Analysis Results:
==========================

Top Holdings: {top_holdings_str}

Risk-Return Metrics:
- Total Return: {metrics.total_return:.2f}%
- Annualized Return: {metrics.annualized_return:.2f}%
- Volatility: {metrics.volatility:.2f}%
- Sharpe Ratio: {metrics.sharpe_ratio:.3f}
- Beta: {metrics.beta:.3f}

Risk Metrics:
- Maximum Drawdown: {metrics.max_drawdown:.2f}%

Diversification:
- Diversification Score: {metrics.diversification_score:.0f}/100
- Risk Level: {metrics.risk_level}

Interpretation:
- Sharpe Ratio: {'Excellent (>2)' if metrics.sharpe_ratio > 2 else 'Good (1-2)' if metrics.sharpe_ratio > 1 else 'Fair (0-1)' if metrics.sharpe_ratio > 0 else 'Poor (<0)'}
- Volatility: {'High (>30%)' if metrics.volatility > 30 else 'Moderate (15-30%)' if metrics.volatility > 15 else 'Low (<15%)'}
- Beta: {'Aggressive (>1.2)' if metrics.beta > 1.2 else 'Market-like (0.8-1.2)' if metrics.beta > 0.8 else 'Defensive (<0.8)'}
"""

    def _create_agent(self):
        """Create the LangChain agent with system prompt"""

        system_prompt = """You are a Portfolio Insight Agent, an expert financial analyst specializing in portfolio risk assessment.

Your role is to:
1. Help users understand their portfolio's risk-return characteristics
2. Explain financial metrics in clear, accessible language
3. Identify strengths and weaknesses in portfolio allocation
4. Provide actionable insights based on quantitative analysis

When analyzing portfolios:
- Use the analyze_portfolio tool to get metrics
- Interpret the numbers in context (market conditions, investor goals)
- Highlight both positive aspects and areas of concern
- Consider diversification using the correlation tool when relevant
- Provide concrete, specific insights rather than generic advice

Communication style:
- Clear and professional
- Balance technical accuracy with accessibility
- Use analogies when helpful
- Always ground insights in the actual data

Remember: You're providing analysis and education, not financial advice or recommendations to buy/sell specific securities."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return agent

    def analyze(self, query: str, chat_history: Optional[List] = None) -> str:
        """
        Analyze a portfolio based on natural language query.

        Args:
            query: User's question or portfolio description
            chat_history: Optional conversation history

        Returns:
            Agent's response with analysis
        """
        try:
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history or []
            })

            output = response["output"]

            # AgentExecutor can return output in different formats
            # Handle list format: [{'text': '...', 'type': 'text', 'index': 0}]
            if isinstance(output, list):
                if len(output) > 0 and isinstance(output[0], dict):
                    if 'text' in output[0]:
                        return output[0]['text']
                # If list but not in expected format, join all items
                return ' '.join(str(item) for item in output)

            # Handle string format (fallback)
            if isinstance(output, str):
                return output

            # Handle any other format
            return str(output)

        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def chat(self, message: str, chat_history: List = None) -> tuple[str, List]:
        """
        Have a conversation with the agent.

        Args:
            message: User message
            chat_history: Conversation history

        Returns:
            Tuple of (response, updated_chat_history)
        """
        if chat_history is None:
            chat_history = []

        response = self.analyze(message, chat_history)

        # Update chat history
        chat_history.append(HumanMessage(content=message))
        chat_history.append(AIMessage(content=response))

        return response, chat_history

    def analyze_live_portfolio(self) -> str:
        """
        Analyze the current portfolio from the service.
        Requires service to be set during initialization.
        """
        if not self.service:
            return "Portfolio service not initialized. Cannot analyze live portfolio."

        if LEGACY_MODE:
            return "Live portfolio analysis requires updated analytics module."

        try:
            # Get current positions
            positions = self.service.get_positions()

            if not positions:
                return "Your portfolio is empty. Start trading to build positions for analysis."

            # Get price history for all positions
            tickers = [p.ticker for p in positions]
            price_history = {}
            sectors = {}

            for ticker in tickers:
                df = self.service.get_price_history(ticker, "3mo")
                if not df.empty:
                    price_history[ticker] = df

                # Get sector info
                info = self.service.get_stock_info(ticker)
                if info:
                    sectors[ticker] = info.get('sector', 'Other')

            # Calculate metrics
            metrics = self.analyzer.calculate_portfolio_metrics(
                positions,
                price_history,
                sectors
            )

            # Generate narrative
            narrative = generate_portfolio_narrative(metrics, positions)

            # Get strengths and weaknesses
            analysis = identify_portfolio_strengths_weaknesses(metrics)

            # Format complete analysis
            result = f"""# Portfolio Analysis

{narrative}

## Key Metrics

**Return Metrics:**
- Total Return: {metrics.total_return:+.2f}%
- Annualized Return: {metrics.annualized_return:+.2f}%

**Risk Metrics:**
- Volatility: {metrics.volatility:.2f}%
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Beta: {metrics.beta:.2f}
- Max Drawdown: {metrics.max_drawdown:.2f}%

**Portfolio Composition:**
- Number of Holdings: {len(positions)}
- Diversification Score: {metrics.diversification_score:.0f}/100
- Risk Level: {metrics.risk_level}

## Strengths ✅
""" + "\n".join([f"- {s}" for s in analysis['strengths']]) + """

## Areas for Improvement ⚠️
""" + "\n".join([f"- {w}" for w in analysis['weaknesses']])

            if metrics.sector_concentration:
                result += "\n\n## Sector Allocation\n"
                for sector, pct in list(metrics.sector_concentration.items())[:5]:
                    result += f"- {sector}: {pct:.1f}%\n"

            return result

        except Exception as e:
            return f"Error analyzing portfolio: {str(e)}"


def main():
    """Example usage of the Portfolio Insight Agent"""
    print("Portfolio Insight Agent")
    print("=" * 50)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")
        return

    # Initialize agent
    use_openai = bool(os.getenv("OPENAI_API_KEY")) and not bool(os.getenv("ANTHROPIC_API_KEY"))
    agent = PortfolioInsightAgent(use_openai=use_openai)

    # Example analysis
    portfolio = "40% AAPL, 30% MSFT, 30% GOOGL"
    print(f"\nAnalyzing portfolio: {portfolio}\n")

    response = agent.analyze(f"Analyze my portfolio: {portfolio}")
    print(response)

    print("\n" + "=" * 50)
    print("\nFollow-up question:")
    response = agent.analyze(
        "What are the main risks in this portfolio?",
        chat_history=[]
    )
    print(response)


if __name__ == "__main__":
    main()
