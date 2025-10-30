"""
Rebalancing Strategist Agent using LangChain.
Provides portfolio optimization and rebalancing recommendations.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from scipy.optimize import minimize

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


class RebalancingStrategist:
    """Utility class for portfolio rebalancing and optimization"""

    @classmethod
    def calculate_optimal_weights(cls, returns: pd.DataFrame, method: str = "sharpe") -> Dict[str, float]:
        """
        Calculate optimal portfolio weights using Modern Portfolio Theory.

        Args:
            returns: DataFrame of historical returns
            method: Optimization method ("sharpe", "min_variance", "risk_parity")

        Returns:
            Dictionary of optimal weights
        """
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(returns.columns)

        if method == "sharpe":
            # Maximize Sharpe ratio
            def neg_sharpe(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -portfolio_return / portfolio_std if portfolio_std > 0 else 0

            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets]

            result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights = result.x

        elif method == "min_variance":
            # Minimize variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets]

            result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights = result.x

        elif method == "risk_parity":
            # Equal risk contribution
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                target_risk = portfolio_vol / num_assets
                return np.sum((risk_contrib - target_risk) ** 2)

            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets]

            result = minimize(risk_parity_objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights = result.x
        else:
            # Equal weight fallback
            optimal_weights = np.array([1.0 / num_assets] * num_assets)

        return {ticker: weight for ticker, weight in zip(returns.columns, optimal_weights)}

    @classmethod
    def generate_rebalancing_plan(cls, current_weights: Dict[str, float], target_weights: Dict[str, float], portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Generate specific rebalancing trades.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            Rebalancing plan with trades
        """
        trades = []
        total_trades_value = 0

        all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))

        for ticker in all_tickers:
            current_weight = current_weights.get(ticker, 0)
            target_weight = target_weights.get(ticker, 0)

            current_value = current_weight * portfolio_value
            target_value = target_weight * portfolio_value
            trade_value = target_value - current_value

            if abs(trade_value) > portfolio_value * 0.005:  # Only trade if >0.5% of portfolio
                action = "BUY" if trade_value > 0 else "SELL"
                trades.append({
                    "ticker": ticker,
                    "action": action,
                    "amount": abs(trade_value),
                    "current_weight": current_weight * 100,
                    "target_weight": target_weight * 100,
                    "weight_change": (target_weight - current_weight) * 100
                })
                total_trades_value += abs(trade_value)

        # Sort by trade size
        trades.sort(key=lambda x: x["amount"], reverse=True)

        return {
            "trades": trades,
            "num_trades": len(trades),
            "total_trade_value": total_trades_value,
            "turnover_pct": (total_trades_value / portfolio_value) * 100
        }

    @classmethod
    def identify_overweight_underweight(cls, current_weights: Dict[str, float], target_weights: Dict[str, float], threshold: float = 0.05) -> Dict[str, List]:
        """
        Identify overweight and underweight positions.

        Args:
            current_weights: Current weights
            target_weights: Target weights (benchmark or target allocation)
            threshold: Minimum deviation to flag (default 5%)

        Returns:
            Dictionary with overweight and underweight lists
        """
        overweight = []
        underweight = []
        aligned = []

        all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            deviation = current - target

            if abs(deviation) > threshold:
                position_info = {
                    "ticker": ticker,
                    "current": current * 100,
                    "target": target * 100,
                    "deviation": deviation * 100
                }

                if deviation > 0:
                    overweight.append(position_info)
                else:
                    underweight.append(position_info)
            else:
                aligned.append({
                    "ticker": ticker,
                    "current": current * 100,
                    "target": target * 100
                })

        overweight.sort(key=lambda x: x["deviation"], reverse=True)
        underweight.sort(key=lambda x: x["deviation"])

        return {
            "overweight": overweight,
            "underweight": underweight,
            "aligned": aligned
        }


class RebalancingStrategistAgent:
    """
    Rebalancing Strategist Agent for portfolio optimization and adjustment recommendations.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.1,
        use_openai: bool = False,
        service=None
    ):
        """Initialize the Rebalancing Strategist Agent"""
        self.analyzer = PortfolioAnalyzer()
        self.strategist = RebalancingStrategist()
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
        """Create tools for rebalancing analysis"""

        def optimize_portfolio_tool(portfolio_and_method: str) -> str:
            """
            Optimize portfolio using Modern Portfolio Theory.
            Input: "portfolio: 40% AAPL, 30% MSFT, 30% GOOGL | method: sharpe"
            Methods: sharpe, min_variance, risk_parity
            """
            try:
                parts = portfolio_and_method.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio: X% TICKER | method: sharpe'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                method = parts[1].replace("method:", "").strip().lower()

                if method not in ["sharpe", "min_variance", "risk_parity"]:
                    return "Invalid method. Use: sharpe, min_variance, or risk_parity"

                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                # Get historical data (simplified - would use real data)
                tickers = list(holdings.keys())
                returns_data = {}
                for ticker in tickers:
                    # Simulate returns (in production, fetch real data)
                    np.random.seed(hash(ticker) % 2**32)
                    returns_data[ticker] = np.random.normal(0.001, 0.02, 252)

                returns_df = pd.DataFrame(returns_data)
                optimal_weights = self.strategist.calculate_optimal_weights(returns_df, method)

                output = f"""
Portfolio Optimization Results ({method.upper()})
================================================

Current Allocation:
"""
                for ticker, weight in holdings.items():
                    output += f"  {ticker}: {weight*100:.1f}%\n"

                output += "\nOptimal Allocation:\n"
                for ticker, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
                    output += f"  {ticker}: {weight*100:.1f}%\n"

                output += "\nWeight Changes:\n"
                for ticker in holdings.keys():
                    current = holdings[ticker]
                    optimal = optimal_weights.get(ticker, 0)
                    change = optimal - current
                    direction = "INCREASE" if change > 0 else "DECREASE" if change < 0 else "NO CHANGE"
                    output += f"  {ticker}: {change*100:+.1f}% ({direction})\n"

                return output

            except Exception as e:
                return f"Error optimizing portfolio: {str(e)}"

        def generate_rebalancing_plan_tool(portfolio_and_target: str) -> str:
            """
            Generate specific rebalancing trades.
            Input: "current: 40% AAPL, 30% MSFT, 30% GOOGL | target: 33% AAPL, 33% MSFT, 34% GOOGL"
            """
            try:
                parts = portfolio_and_target.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'current: X% TICKER | target: Y% TICKER'"

                current_str = parts[0].replace("current:", "").strip()
                target_str = parts[1].replace("target:", "").strip()

                current_weights = self.analyzer.parse_portfolio(current_str)
                target_weights = self.analyzer.parse_portfolio(target_str)

                if not current_weights or not target_weights:
                    return "Could not parse portfolios."

                plan = self.strategist.generate_rebalancing_plan(current_weights, target_weights)

                output = f"""
Rebalancing Plan
================

Total Trades: {plan['num_trades']}
Portfolio Turnover: {plan['turnover_pct']:.1f}%
Total Trade Value: ${plan['total_trade_value']:,.2f}

Specific Trades:
"""
                for i, trade in enumerate(plan['trades'], 1):
                    output += f"\n{i}. {trade['action']} {trade['ticker']}\n"
                    output += f"   Amount: ${trade['amount']:,.2f}\n"
                    output += f"   Current Weight: {trade['current_weight']:.1f}%\n"
                    output += f"   Target Weight: {trade['target_weight']:.1f}%\n"
                    output += f"   Change: {trade['weight_change']:+.1f}%\n"

                return output

            except Exception as e:
                return f"Error generating rebalancing plan: {str(e)}"

        def identify_drift_tool(portfolio_and_benchmark: str) -> str:
            """
            Identify portfolio drift from target allocation.
            Input: "portfolio: 45% AAPL, 35% MSFT, 20% GOOGL | benchmark: 33% AAPL, 33% MSFT, 34% GOOGL"
            """
            try:
                parts = portfolio_and_benchmark.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio: X% TICKER | benchmark: Y% TICKER'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                benchmark_str = parts[1].replace("benchmark:", "").strip()

                current = self.analyzer.parse_portfolio(portfolio_str)
                target = self.analyzer.parse_portfolio(benchmark_str)

                if not current or not target:
                    return "Could not parse inputs."

                analysis = self.strategist.identify_overweight_underweight(current, target)

                output = """
Portfolio Drift Analysis
========================

"""
                if analysis['overweight']:
                    output += "OVERWEIGHT Positions (Reduce):\n"
                    for pos in analysis['overweight']:
                        output += f"  {pos['ticker']}: {pos['current']:.1f}% (target: {pos['target']:.1f}%, {pos['deviation']:+.1f}% over)\n"
                    output += "\n"

                if analysis['underweight']:
                    output += "UNDERWEIGHT Positions (Increase):\n"
                    for pos in analysis['underweight']:
                        output += f"  {pos['ticker']}: {pos['current']:.1f}% (target: {pos['target']:.1f}%, {abs(pos['deviation']):.1f}% under)\n"
                    output += "\n"

                if analysis['aligned']:
                    output += "ALIGNED Positions (No action needed):\n"
                    for pos in analysis['aligned']:
                        output += f"  {pos['ticker']}: {pos['current']:.1f}% (target: {pos['target']:.1f}%)\n"

                return output

            except Exception as e:
                return f"Error identifying drift: {str(e)}"

        tools = [
            Tool(
                name="optimize_portfolio",
                func=optimize_portfolio_tool,
                description=(
                    "Optimizes portfolio weights using Modern Portfolio Theory. "
                    "Input: 'portfolio: 40% AAPL, 30% MSFT | method: sharpe'. "
                    "Methods: sharpe (max Sharpe ratio), min_variance (minimize risk), risk_parity (equal risk contribution)."
                )
            ),
            Tool(
                name="generate_rebalancing_plan",
                func=generate_rebalancing_plan_tool,
                description=(
                    "Generates specific buy/sell trades to rebalance portfolio. "
                    "Input: 'current: 40% AAPL, 30% MSFT | target: 33% AAPL, 33% MSFT'. "
                    "Returns exact trades with dollar amounts."
                )
            ),
            Tool(
                name="identify_drift",
                func=identify_drift_tool,
                description=(
                    "Identifies portfolio drift from target allocation. "
                    "Input: 'portfolio: 45% AAPL, 35% MSFT | benchmark: 33% AAPL, 33% MSFT'. "
                    "Shows overweight, underweight, and aligned positions."
                )
            )
        ]

        return tools

    def _create_agent(self):
        """Create agent with system prompt"""

        system_prompt = """You are a Rebalancing Strategist Agent, specializing in portfolio optimization and rebalancing recommendations.

Your role is to:
1. Optimize portfolio allocations using Modern Portfolio Theory (MPT)
2. Generate specific rebalancing trades (buy/sell recommendations)
3. Identify portfolio drift from target allocations
4. Recommend risk reduction strategies without sacrificing too much return
5. Apply tax-aware rebalancing when relevant

Optimization Methods:
- SHARPE: Maximizes risk-adjusted returns (Sharpe ratio) - best for growth-focused investors
- MIN_VARIANCE: Minimizes portfolio volatility - best for conservative investors
- RISK_PARITY: Equal risk contribution from each asset - best for balanced diversification

When providing recommendations:
- Be specific with exact percentages and dollar amounts
- Explain the rationale behind suggested changes
- Consider transaction costs and tax implications
- Prioritize the most impactful trades
- Note when portfolio is already well-balanced

Communication style:
- Actionable and specific (not just "rebalance", but "sell 5% of AAPL")
- Clear about trade-offs (reducing risk often means lower expected return)
- Educational about MPT concepts
- Practical about implementation

Remember: You provide optimization recommendations, not guarantees of future performance. All models have limitations."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return agent

    def analyze(self, query: str, chat_history: Optional[List] = None) -> str:
        """Analyze rebalancing query"""
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
    print("Rebalancing Strategist Agent")
    print("=" * 50)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set API key in .env file")
        return

    agent = RebalancingStrategistAgent()

    print("\n[Portfolio Optimization]\n")
    response = agent.analyze(
        "Optimize my portfolio to maximize Sharpe ratio: 40% AAPL, 30% MSFT, 30% GOOGL"
    )
    print(response)


if __name__ == "__main__":
    main()
