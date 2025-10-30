"""
Temporal Intelligence Agent using LangChain.
Analyzes time-based risk patterns and investment horizons.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.portfolio_analytics import PortfolioAnalyzer

load_dotenv()


class TemporalAnalytics:
    """Utility class for time-based analysis"""

    @classmethod
    def calculate_rolling_metrics(cls, returns: pd.Series, window_days: int = 252) -> Dict[str, Any]:
        """
        Calculate rolling risk metrics over time.

        Args:
            returns: Time series of returns
            window_days: Rolling window size

        Returns:
            Dictionary of rolling metrics
        """
        rolling_vol = returns.rolling(window=window_days).std() * np.sqrt(252) * 100
        rolling_sharpe = (returns.rolling(window=window_days).mean() * 252) / (returns.rolling(window=window_days).std() * np.sqrt(252))

        # Calculate drawdown over time
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100

        results = {
            "current_volatility": rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0,
            "avg_volatility": rolling_vol.mean() if len(rolling_vol) > 0 else 0,
            "min_volatility": rolling_vol.min() if len(rolling_vol) > 0 else 0,
            "max_volatility": rolling_vol.max() if len(rolling_vol) > 0 else 0,
            "current_sharpe": rolling_sharpe.iloc[-1] if len(rolling_sharpe) > 0 else 0,
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0,
            "max_drawdown_ever": drawdown.min() if len(drawdown) > 0 else 0,
            "volatility_trend": "Increasing" if rolling_vol.iloc[-1] > rolling_vol.iloc[-30] else "Decreasing" if len(rolling_vol) > 30 else "Stable"
        }

        return results

    @classmethod
    def analyze_time_horizons(cls, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze portfolio suitability for different investment horizons.

        Args:
            metrics: Portfolio metrics (volatility, Sharpe, etc.)

        Returns:
            Time horizon analysis
        """
        volatility = metrics.get("volatility", 20)
        sharpe = metrics.get("sharpe_ratio", 1.0)
        max_drawdown = metrics.get("max_drawdown", -20)

        horizons = {}

        # Short-term (< 1 year)
        if volatility > 25:
            short_term_suitability = "LOW"
            short_term_reason = "High volatility increases short-term loss probability"
        elif abs(max_drawdown) > 20:
            short_term_suitability = "LOW"
            short_term_reason = "Large drawdowns pose risk for short-term needs"
        else:
            short_term_suitability = "MEDIUM"
            short_term_reason = "Moderate risk acceptable for 6-12 month horizon"

        horizons["short_term"] = {
            "horizon": "< 1 year",
            "suitability": short_term_suitability,
            "reason": short_term_reason
        }

        # Medium-term (1-5 years)
        if sharpe > 1.0 and volatility < 25:
            medium_term_suitability = "HIGH"
            medium_term_reason = "Good risk-adjusted returns suitable for medium horizon"
        elif volatility > 30:
            medium_term_suitability = "MEDIUM"
            medium_term_reason = "Higher volatility acceptable with 3-5 year horizon"
        else:
            medium_term_suitability = "HIGH"
            medium_term_reason = "Time diversification reduces volatility impact"

        horizons["medium_term"] = {
            "horizon": "1-5 years",
            "suitability": medium_term_suitability,
            "reason": medium_term_reason
        }

        # Long-term (5+ years)
        if sharpe > 0.5:
            long_term_suitability = "HIGH"
            long_term_reason = "Long horizon allows volatility to smooth out"
        elif volatility > 40:
            long_term_suitability = "MEDIUM"
            long_term_reason = "Extreme volatility still a concern even long-term"
        else:
            long_term_suitability = "HIGH"
            long_term_reason = "Long-term growth potential outweighs short-term volatility"

        horizons["long_term"] = {
            "horizon": "5+ years",
            "suitability": long_term_suitability,
            "reason": long_term_reason
        }

        # Recommended horizon
        suitability_scores = {
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1
        }

        best_horizon = max(horizons.items(), key=lambda x: suitability_scores[x[1]["suitability"]])

        return {
            "horizons": horizons,
            "recommended_horizon": best_horizon[0].replace("_", " ").title(),
            "recommendation_reason": best_horizon[1]["reason"]
        }

    @classmethod
    def analyze_risk_evolution(cls, historical_volatility: List[float], periods: List[str]) -> Dict[str, Any]:
        """
        Analyze how risk has evolved over time.

        Args:
            historical_volatility: List of volatility values over time
            periods: Corresponding time period labels

        Returns:
            Risk evolution analysis
        """
        if len(historical_volatility) < 2:
            return {"trend": "Unknown", "analysis": "Insufficient data"}

        # Calculate trend
        volatility_changes = [historical_volatility[i] - historical_volatility[i-1]
                              for i in range(1, len(historical_volatility))]

        avg_change = np.mean(volatility_changes)
        recent_vol = historical_volatility[-1]
        historical_avg = np.mean(historical_volatility)

        if avg_change > 1:
            trend = "Increasing"
            trend_description = f"Risk has been increasing (avg change: +{avg_change:.1f}% per period)"
        elif avg_change < -1:
            trend = "Decreasing"
            trend_description = f"Risk has been decreasing (avg change: {avg_change:.1f}% per period)"
        else:
            trend = "Stable"
            trend_description = f"Risk has remained relatively stable (avg change: {avg_change:+.1f}%)"

        comparison = "above" if recent_vol > historical_avg else "below"
        comparison_pct = abs(recent_vol - historical_avg) / historical_avg * 100

        return {
            "trend": trend,
            "trend_description": trend_description,
            "current_vs_historical": f"Current volatility is {comparison_pct:.1f}% {comparison} historical average",
            "current_vol": recent_vol,
            "historical_avg": historical_avg,
            "min_vol": min(historical_volatility),
            "max_vol": max(historical_volatility)
        }


class TemporalIntelligenceAgent:
    """
    Temporal Intelligence Agent for time-based risk analysis.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.1,
        use_openai: bool = False,
        service=None
    ):
        """Initialize the Temporal Intelligence Agent"""
        self.analyzer = PortfolioAnalyzer()
        self.temporal = TemporalAnalytics()
        self.service = service
        self.use_openai = use_openai

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

        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def _create_tools(self) -> List[Tool]:
        """Create tools for temporal analysis"""

        def analyze_time_horizon_tool(portfolio_and_horizon: str) -> str:
            """
            Analyze portfolio suitability for investment horizon.
            Input: "portfolio: 40% AAPL, 30% MSFT, 30% GOOGL | horizon: short"
            Horizons: short (< 1 year), medium (1-5 years), long (5+ years), all
            """
            try:
                parts = portfolio_and_horizon.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio: X% TICKER | horizon: short/medium/long/all'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                horizon = parts[1].replace("horizon:", "").strip().lower()

                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                metrics = self.analyzer.analyze_portfolio(holdings)

                metrics_dict = {
                    "volatility": metrics.volatility if hasattr(metrics, 'volatility') else 20,
                    "sharpe_ratio": metrics.sharpe_ratio if hasattr(metrics, 'sharpe_ratio') else 1.0,
                    "max_drawdown": metrics.max_drawdown if hasattr(metrics, 'max_drawdown') else -20,
                }

                analysis = self.temporal.analyze_time_horizons(metrics_dict)

                if horizon == "all":
                    output = """
Time Horizon Analysis
=====================

"""
                    for horizon_key, data in analysis["horizons"].items():
                        output += f"\n{horizon_key.replace('_', ' ').upper()} ({data['horizon']}):\n"
                        output += f"  Suitability: {data['suitability']}\n"
                        output += f"  Reason: {data['reason']}\n"

                    output += f"\n\nRECOMMENDED HORIZON: {analysis['recommended_horizon']}\n"
                    output += f"Reason: {analysis['recommendation_reason']}"

                    return output
                else:
                    horizon_key = f"{horizon}_term"
                    if horizon_key not in analysis["horizons"]:
                        return f"Invalid horizon: {horizon}. Use: short, medium, long, or all"

                    data = analysis["horizons"][horizon_key]
                    return f"""
{horizon.upper()}-TERM Analysis ({data['horizon']})
{'=' * 50}

Suitability: {data['suitability']}

Reason: {data['reason']}

Portfolio Metrics:
- Volatility: {metrics_dict['volatility']:.1f}%
- Sharpe Ratio: {metrics_dict['sharpe_ratio']:.2f}
- Max Drawdown: {metrics_dict['max_drawdown']:.1f}%
"""

            except Exception as e:
                return f"Error analyzing time horizon: {str(e)}"

        def compare_time_periods_tool(portfolio_str: str) -> str:
            """
            Compare portfolio performance across different time periods.
            Input: "portfolio: 40% AAPL, 30% MSFT, 30% GOOGL"
            Analyzes 1-month, 3-month, 6-month, 1-year, 5-year volatility
            """
            try:
                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                # Simulate different time period volatilities (in production, use real data)
                periods = {
                    "1-month": np.random.uniform(15, 25),
                    "3-month": np.random.uniform(18, 28),
                    "6-month": np.random.uniform(17, 27),
                    "1-year": np.random.uniform(16, 26),
                    "5-year": np.random.uniform(19, 29)
                }

                output = """
Multi-Period Volatility Analysis
=================================

Volatility by Time Period:
"""
                for period, vol in periods.items():
                    risk_level = "High" if vol > 25 else "Moderate" if vol > 18 else "Low"
                    output += f"\n{period}: {vol:.1f}% ({risk_level} risk)\n"

                # Trend analysis
                values = list(periods.values())
                if values[-1] > values[0]:
                    trend = "Volatility has INCREASED over longer time periods"
                elif values[-1] < values[0]:
                    trend = "Volatility has DECREASED over longer time periods"
                else:
                    trend = "Volatility has remained STABLE across time periods"

                output += f"\n\nTrend: {trend}\n"

                output += "\nInterpretation:\n"
                output += "- Short-term volatility indicates recent market conditions\n"
                output += "- Long-term volatility shows historical stability\n"
                output += "- Increasing volatility suggests rising risk environment\n"

                return output

            except Exception as e:
                return f"Error comparing time periods: {str(e)}"

        def analyze_risk_trend_tool(volatility_data: str) -> str:
            """
            Analyze risk trend over time.
            Input: "volatility: 15, 18, 22, 25, 20 | periods: Jan, Feb, Mar, Apr, May"
            """
            try:
                parts = volatility_data.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'volatility: 15, 18, 22 | periods: Jan, Feb, Mar'"

                vol_str = parts[0].replace("volatility:", "").strip()
                period_str = parts[1].replace("periods:", "").strip()

                volatilities = [float(v.strip()) for v in vol_str.split(",")]
                periods = [p.strip() for p in period_str.split(",")]

                if len(volatilities) != len(periods):
                    return "Number of volatility values must match number of periods"

                analysis = self.temporal.analyze_risk_evolution(volatilities, periods)

                output = f"""
Risk Evolution Analysis
=======================

Trend: {analysis['trend']}
{analysis['trend_description']}

Current vs Historical:
{analysis['current_vs_historical']}

Statistics:
- Current Volatility: {analysis['current_vol']:.1f}%
- Historical Average: {analysis['historical_avg']:.1f}%
- Minimum: {analysis['min_vol']:.1f}%
- Maximum: {analysis['max_vol']:.1f}%

Period-by-Period:
"""
                for i, (period, vol) in enumerate(zip(periods, volatilities)):
                    change = "" if i == 0 else f" ({volatilities[i] - volatilities[i-1]:+.1f}%)"
                    output += f"  {period}: {vol:.1f}%{change}\n"

                return output

            except Exception as e:
                return f"Error analyzing risk trend: {str(e)}"

        tools = [
            Tool(
                name="analyze_time_horizon",
                func=analyze_time_horizon_tool,
                description=(
                    "Analyzes portfolio suitability for different investment horizons. "
                    "Input: 'portfolio: 40% AAPL, 30% MSFT | horizon: short/medium/long/all'. "
                    "Returns suitability assessment for short (<1yr), medium (1-5yr), or long (5+yr) horizons."
                )
            ),
            Tool(
                name="compare_time_periods",
                func=compare_time_periods_tool,
                description=(
                    "Compares portfolio volatility across different time periods. "
                    "Input: 'portfolio: 40% AAPL, 30% MSFT, 30% GOOGL'. "
                    "Analyzes 1-month, 3-month, 6-month, 1-year, and 5-year volatility."
                )
            ),
            Tool(
                name="analyze_risk_trend",
                func=analyze_risk_trend_tool,
                description=(
                    "Analyzes how portfolio risk has evolved over time. "
                    "Input: 'volatility: 15, 18, 22, 25 | periods: Jan, Feb, Mar, Apr'. "
                    "Shows trend direction and statistics."
                )
            )
        ]

        return tools

    def _create_agent(self):
        """Create agent with system prompt"""

        system_prompt = """You are a Temporal Intelligence Agent, specializing in time-based risk analysis and investment horizon planning.

Your role is to:
1. Analyze how portfolio risk changes across different time periods
2. Assess portfolio suitability for different investment horizons (short/medium/long-term)
3. Identify risk trends and patterns over time
4. Perform rolling risk metric analysis
5. Provide time-horizon appropriate recommendations

Investment Horizons:
- SHORT-TERM (< 1 year): Focus on capital preservation, minimize volatility
- MEDIUM-TERM (1-5 years): Balance growth and stability, moderate risk acceptable
- LONG-TERM (5+ years): Growth-focused, volatility smooths out over time

When analyzing temporal patterns:
- Distinguish between short-term volatility spikes and structural changes
- Consider market cycles (bull/bear markets, recessions)
- Explain why time horizon matters for risk tolerance
- Identify if current risk is elevated vs historical norms
- Note seasonal or cyclical patterns

Communication style:
- Clear about the time dimension of risk
- Educational about time diversification benefits
- Specific with time frames and dates
- Balanced about short vs long-term perspectives

Remember: Time is a crucial dimension in risk assessment. The same portfolio may be appropriate for a 10-year horizon but inappropriate for a 1-year horizon."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return agent

    def analyze(self, query: str, chat_history: Optional[List] = None) -> str:
        """Analyze temporal query"""
        try:
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history or []
            })
            return self._extract_output(response["output"])
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def _extract_output(self, output: Any) -> str:
        """Extract string output"""
        if isinstance(output, list):
            if len(output) > 0 and isinstance(output[0], dict) and 'text' in output[0]:
                return output[0]['text']
            return ' '.join(str(item) for item in output)
        return str(output)


def main():
    """Example usage"""
    print("Temporal Intelligence Agent")
    print("=" * 50)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set API key in .env file")
        return

    agent = TemporalIntelligenceAgent()

    print("\n[Time Horizon Analysis]\n")
    response = agent.analyze(
        "Is this portfolio suitable for a 10-year investment horizon? Portfolio: 40% AAPL, 30% MSFT, 30% GOOGL"
    )
    print(response)


if __name__ == "__main__":
    main()
