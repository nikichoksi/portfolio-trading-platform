"""
Comparative Analytics Agent using LangChain.
Provides cross-sectional analysis and benchmarking.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
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


class ComparativeAnalytics:
    """Utility class for comparative analysis"""

    # Benchmark portfolios
    BENCHMARKS = {
        "sp500": {"name": "S&P 500", "annual_return": 10.0, "volatility": 18.0, "sharpe": 0.55, "beta": 1.0},
        "nasdaq": {"name": "NASDAQ-100", "annual_return": 12.0, "volatility": 22.0, "sharpe": 0.54, "beta": 1.15},
        "conservative_60_40": {"name": "60/40 Stocks/Bonds", "annual_return": 7.5, "volatility": 10.0, "sharpe": 0.75, "beta": 0.6},
        "aggressive_growth": {"name": "Aggressive Growth", "annual_return": 14.0, "volatility": 28.0, "sharpe": 0.50, "beta": 1.35},
    }

    @classmethod
    def compare_portfolios(cls, portfolio1: Dict[str, Any], portfolio2: Dict[str, Any], label1: str = "Portfolio A", label2: str = "Portfolio B") -> Dict[str, Any]:
        """
        Compare two portfolios side-by-side.

        Args:
            portfolio1: First portfolio metrics
            portfolio2: Second portfolio metrics
            label1: Label for first portfolio
            label2: Label for second portfolio

        Returns:
            Comparison results
        """
        comparison = {
            "labels": [label1, label2],
            "metrics": {},
            "winner": {},
            "summary": ""
        }

        # Compare key metrics
        metrics_to_compare = ["volatility", "beta", "sharpe_ratio", "annual_return", "max_drawdown"]

        for metric in metrics_to_compare:
            val1 = portfolio1.get(metric, 0)
            val2 = portfolio2.get(metric, 0)

            # Determine winner (lower is better for risk metrics, higher for returns)
            if metric in ["volatility", "beta", "max_drawdown"]:
                winner = label1 if val1 < val2 else label2
                diff_pct = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            else:  # sharpe_ratio, annual_return
                winner = label1 if val1 > val2 else label2
                diff_pct = ((val1 - val2) / val2 * 100) if val2 != 0 else 0

            comparison["metrics"][metric] = {
                label1: val1,
                label2: val2,
                "difference_pct": abs(diff_pct),
                "winner": winner
            }

        # Overall assessment
        wins1 = sum(1 for m in comparison["metrics"].values() if m["winner"] == label1)
        wins2 = sum(1 for m in comparison["metrics"].values() if m["winner"] == label2)

        if wins1 > wins2:
            comparison["overall_winner"] = label1
            comparison["summary"] = f"{label1} outperforms on {wins1} out of {len(metrics_to_compare)} key metrics."
        elif wins2 > wins1:
            comparison["overall_winner"] = label2
            comparison["summary"] = f"{label2} outperforms on {wins2} out of {len(metrics_to_compare)} key metrics."
        else:
            comparison["overall_winner"] = "Tie"
            comparison["summary"] = "Both portfolios perform similarly across key metrics."

        return comparison

    @classmethod
    def benchmark_against_index(cls, portfolio_metrics: Dict[str, Any], benchmark_key: str = "sp500") -> Dict[str, Any]:
        """
        Benchmark portfolio against market index.

        Args:
            portfolio_metrics: Portfolio metrics
            benchmark_key: Benchmark identifier

        Returns:
            Benchmarking results
        """
        if benchmark_key not in cls.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_key}")

        benchmark = cls.BENCHMARKS[benchmark_key]

        results = {
            "portfolio_name": "Your Portfolio",
            "benchmark_name": benchmark["name"],
            "metrics": {},
            "relative_performance": {}
        }

        # Compare metrics
        volatility_diff = portfolio_metrics.get("volatility", 0) - benchmark["volatility"]
        return_diff = portfolio_metrics.get("annual_return", 0) - benchmark["annual_return"]
        sharpe_diff = portfolio_metrics.get("sharpe_ratio", 0) - benchmark["sharpe"]
        beta_diff = portfolio_metrics.get("beta", 0) - benchmark["beta"]

        results["metrics"] = {
            "volatility": {
                "portfolio": portfolio_metrics.get("volatility", 0),
                "benchmark": benchmark["volatility"],
                "difference": volatility_diff,
                "assessment": "Higher risk" if volatility_diff > 0 else "Lower risk"
            },
            "annual_return": {
                "portfolio": portfolio_metrics.get("annual_return", 0),
                "benchmark": benchmark["annual_return"],
                "difference": return_diff,
                "assessment": "Outperforming" if return_diff > 0 else "Underperforming"
            },
            "sharpe_ratio": {
                "portfolio": portfolio_metrics.get("sharpe_ratio", 0),
                "benchmark": benchmark["sharpe"],
                "difference": sharpe_diff,
                "assessment": "Better risk-adjusted returns" if sharpe_diff > 0 else "Worse risk-adjusted returns"
            },
            "beta": {
                "portfolio": portfolio_metrics.get("beta", 0),
                "benchmark": benchmark["beta"],
                "difference": beta_diff,
                "assessment": "More volatile than market" if beta_diff > 0 else "Less volatile than market"
            }
        }

        # Overall assessment
        if return_diff > 2 and sharpe_diff > 0:
            results["overall_assessment"] = "STRONG: Outperforming benchmark with better risk-adjusted returns"
        elif return_diff > 0 and sharpe_diff > 0:
            results["overall_assessment"] = "GOOD: Outperforming benchmark with reasonable risk"
        elif return_diff < 0 and sharpe_diff < 0:
            results["overall_assessment"] = "WEAK: Underperforming benchmark with worse risk profile"
        else:
            results["overall_assessment"] = "MIXED: Some metrics outperform, others underperform"

        return results

    @classmethod
    def compare_sectors(cls, sector1_metrics: Dict[str, float], sector2_metrics: Dict[str, float], sector1_name: str, sector2_name: str) -> Dict[str, Any]:
        """
        Compare risk characteristics of two sectors.

        Args:
            sector1_metrics: Metrics for sector 1
            sector2_metrics: Metrics for sector 2
            sector1_name: Name of sector 1
            sector2_name: Name of sector 2

        Returns:
            Sector comparison results
        """
        comparison = {
            "sectors": [sector1_name, sector2_name],
            "metrics_comparison": {},
            "recommendation": ""
        }

        for metric in ["volatility", "beta", "sharpe_ratio"]:
            val1 = sector1_metrics.get(metric, 0)
            val2 = sector2_metrics.get(metric, 0)

            comparison["metrics_comparison"][metric] = {
                sector1_name: val1,
                sector2_name: val2,
                "difference": val1 - val2
            }

        # Generate recommendation
        vol_diff = sector1_metrics.get("volatility", 0) - sector2_metrics.get("volatility", 0)
        if vol_diff < -5:
            comparison["recommendation"] = f"{sector1_name} is significantly less risky than {sector2_name}"
        elif vol_diff > 5:
            comparison["recommendation"] = f"{sector2_name} is significantly less risky than {sector1_name}"
        else:
            comparison["recommendation"] = f"Both sectors have similar risk profiles"

        return comparison


class ComparativeAnalyticsAgent:
    """
    Comparative Analytics Agent for cross-sectional analysis and benchmarking.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.1,
        use_openai: bool = False,
        service=None
    ):
        """Initialize the Comparative Analytics Agent"""
        self.analyzer = PortfolioAnalyzer()
        self.comparator = ComparativeAnalytics()
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
        """Create tools for comparative analysis"""

        def compare_two_portfolios_tool(portfolios: str) -> str:
            """
            Compare two portfolios side-by-side.
            Input: "portfolio1: 40% AAPL, 30% MSFT, 30% GOOGL | portfolio2: 50% VOO, 50% BND"
            """
            try:
                parts = portfolios.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio1: X% TICKER | portfolio2: Y% TICKER'"

                portfolio1_str = parts[0].replace("portfolio1:", "").strip()
                portfolio2_str = parts[1].replace("portfolio2:", "").strip()

                holdings1 = self.analyzer.parse_portfolio(portfolio1_str)
                holdings2 = self.analyzer.parse_portfolio(portfolio2_str)

                if not holdings1 or not holdings2:
                    return "Could not parse portfolios."

                metrics1 = self.analyzer.analyze_portfolio(holdings1)
                metrics2 = self.analyzer.analyze_portfolio(holdings2)

                # Convert to dict format
                dict1 = {
                    "volatility": metrics1.volatility if hasattr(metrics1, 'volatility') else 0,
                    "beta": metrics1.beta if hasattr(metrics1, 'beta') else 0,
                    "sharpe_ratio": metrics1.sharpe_ratio if hasattr(metrics1, 'sharpe_ratio') else 0,
                    "annual_return": metrics1.annualized_return if hasattr(metrics1, 'annualized_return') else 0,
                    "max_drawdown": metrics1.max_drawdown if hasattr(metrics1, 'max_drawdown') else 0,
                }

                dict2 = {
                    "volatility": metrics2.volatility if hasattr(metrics2, 'volatility') else 0,
                    "beta": metrics2.beta if hasattr(metrics2, 'beta') else 0,
                    "sharpe_ratio": metrics2.sharpe_ratio if hasattr(metrics2, 'sharpe_ratio') else 0,
                    "annual_return": metrics2.annualized_return if hasattr(metrics2, 'annualized_return') else 0,
                    "max_drawdown": metrics2.max_drawdown if hasattr(metrics2, 'max_drawdown') else 0,
                }

                comparison = self.comparator.compare_portfolios(dict1, dict2, "Portfolio 1", "Portfolio 2")

                output = """
Portfolio Comparison
====================

"""
                for metric, data in comparison["metrics"].items():
                    output += f"\n{metric.replace('_', ' ').title()}:\n"
                    output += f"  Portfolio 1: {data['Portfolio 1']:.2f}\n"
                    output += f"  Portfolio 2: {data['Portfolio 2']:.2f}\n"
                    output += f"  Winner: {data['winner']} (by {data['difference_pct']:.1f}%)\n"

                output += f"\n\nOverall Assessment:\n{comparison['summary']}\n"
                output += f"Overall Winner: {comparison['overall_winner']}"

                return output

            except Exception as e:
                return f"Error comparing portfolios: {str(e)}"

        def benchmark_portfolio_tool(portfolio_and_benchmark: str) -> str:
            """
            Benchmark portfolio against market index.
            Input: "portfolio: 40% AAPL, 30% MSFT, 30% GOOGL | benchmark: sp500"
            Benchmarks: sp500, nasdaq, conservative_60_40, aggressive_growth
            """
            try:
                parts = portfolio_and_benchmark.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio: X% TICKER | benchmark: sp500'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                benchmark_key = parts[1].replace("benchmark:", "").strip().lower()

                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                metrics = self.analyzer.analyze_portfolio(holdings)

                # Convert to dict
                metrics_dict = {
                    "volatility": metrics.volatility if hasattr(metrics, 'volatility') else 0,
                    "beta": metrics.beta if hasattr(metrics, 'beta') else 0,
                    "sharpe_ratio": metrics.sharpe_ratio if hasattr(metrics, 'sharpe_ratio') else 0,
                    "annual_return": metrics.annualized_return if hasattr(metrics, 'annualized_return') else 0,
                }

                results = self.comparator.benchmark_against_index(metrics_dict, benchmark_key)

                output = f"""
Benchmark Analysis: {results['benchmark_name']}
{'=' * 50}

"""
                for metric, data in results["metrics"].items():
                    output += f"\n{metric.replace('_', ' ').title()}:\n"
                    output += f"  Your Portfolio: {data['portfolio']:.2f}\n"
                    output += f"  {results['benchmark_name']}: {data['benchmark']:.2f}\n"
                    output += f"  Difference: {data['difference']:+.2f}\n"
                    output += f"  Assessment: {data['assessment']}\n"

                output += f"\n\nOverall Assessment:\n{results['overall_assessment']}"

                return output

            except Exception as e:
                return f"Error benchmarking: {str(e)}"

        def list_benchmarks_tool(dummy: str = "") -> str:
            """
            List available benchmarks for comparison.
            Input: any string (not used)
            """
            output = """
Available Benchmarks
====================

"""
            for key, benchmark in self.comparator.BENCHMARKS.items():
                output += f"\n{key.upper()}: {benchmark['name']}\n"
                output += f"  Annual Return: {benchmark['annual_return']:.1f}%\n"
                output += f"  Volatility: {benchmark['volatility']:.1f}%\n"
                output += f"  Sharpe Ratio: {benchmark['sharpe']:.2f}\n"
                output += f"  Beta: {benchmark['beta']:.2f}\n"

            return output

        tools = [
            Tool(
                name="compare_two_portfolios",
                func=compare_two_portfolios_tool,
                description=(
                    "Compares two portfolios side-by-side across key risk metrics. "
                    "Input: 'portfolio1: 40% AAPL, 30% MSFT | portfolio2: 50% VOO, 50% BND'. "
                    "Returns detailed comparison with winner for each metric."
                )
            ),
            Tool(
                name="benchmark_portfolio",
                func=benchmark_portfolio_tool,
                description=(
                    "Benchmarks portfolio against market indices. "
                    "Input: 'portfolio: 40% AAPL, 30% MSFT | benchmark: sp500'. "
                    "Available benchmarks: sp500, nasdaq, conservative_60_40, aggressive_growth."
                )
            ),
            Tool(
                name="list_benchmarks",
                func=list_benchmarks_tool,
                description=(
                    "Lists all available benchmarks with their characteristics. "
                    "Input: any string (not used). "
                    "Returns list of benchmarks with returns, volatility, Sharpe, and beta."
                )
            )
        ]

        return tools

    def _create_agent(self):
        """Create agent with system prompt"""

        system_prompt = """You are a Comparative Analytics Agent, specializing in cross-sectional portfolio analysis and benchmarking.

Your role is to:
1. Compare multiple portfolios or investment options side-by-side
2. Benchmark portfolios against market indices (S&P 500, NASDAQ, etc.)
3. Analyze sector-level risk differences
4. Evaluate style comparisons (growth vs value, large vs small cap)
5. Provide peer portfolio analysis

Available Benchmarks:
- sp500: S&P 500 index (broad US market)
- nasdaq: NASDAQ-100 (tech-heavy)
- conservative_60_40: Traditional 60% stocks, 40% bonds
- aggressive_growth: High-growth portfolio

When providing comparisons:
- Be specific about which metrics favor which option
- Explain the practical implications of differences
- Consider investor context (time horizon, risk tolerance)
- Note when differences are statistically significant
- Provide balanced assessment, not just "pick the winner"

Communication style:
- Objective and data-driven
- Clear about trade-offs between options
- Educational about why metrics differ
- Practical about which differences matter

Remember: Comparison is about understanding trade-offs, not declaring absolute winners. Different portfolios suit different investors."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return agent

    def analyze(self, query: str, chat_history: Optional[List] = None) -> str:
        """Analyze comparative query"""
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
    print("Comparative Analytics Agent")
    print("=" * 50)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set API key in .env file")
        return

    agent = ComparativeAnalyticsAgent()

    print("\n[Benchmark Against S&P 500]\n")
    response = agent.analyze(
        "How does my portfolio compare to the S&P 500? Portfolio: 40% AAPL, 30% MSFT, 30% GOOGL"
    )
    print(response)


if __name__ == "__main__":
    main()
