import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
from typing import TypedDict, Annotated, List, Dict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import operator
import os

LLM_PROVIDER = "groq"  

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        max_tokens=8000
    )
elif LLM_PROVIDER == "openai":
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
elif LLM_PROVIDER == "anthropic":
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

class PortfolioState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    portfolio_data: Dict
    analysis_result: str

@tool
def analyze_portfolio(portfolio_input: str) -> str:
    """
    Analyze a stock portfolio and calculate risk-return metrics.
    
    Args:
        portfolio_input: Portfolio specification in format "TICKER1:WEIGHT1,TICKER2:WEIGHT2" 
                        (e.g., "AAPL:0.4,MSFT:0.3,GOOGL:0.3") or just "TICKER1,TICKER2,TICKER3" for equal weights
    
    Returns:
        JSON string containing portfolio metrics and analysis
    """
    
    try:
        # Parse input
        tickers, weights = parse_portfolio_input(portfolio_input)
        
        if not tickers:
            return json.dumps({"error": "No valid ticker symbols found"})
        
        # Fetch data
        analyzer = PortfolioAnalyzer()
        metrics = analyzer.calculate_portfolio_metrics(tickers, weights)
        
        if "error" in metrics:
            return json.dumps(metrics)
        
        # Format results
        result = {
            "portfolio_composition": {ticker: f"{weight*100:.1f}%" for ticker, weight in weights.items()},
            "risk_metrics": {
                "annual_return": f"{metrics['annual_return']*100:.2f}%",
                "annual_volatility": f"{metrics['annual_volatility']*100:.2f}%",
                "sharpe_ratio": f"{metrics['sharpe_ratio']:.2f}",
                "beta": f"{metrics['beta']:.2f}" if metrics['beta'] else "N/A",
                "max_drawdown": f"{metrics['max_drawdown']*100:.2f}%",
            },
            "diversification": {
                "number_of_holdings": len(tickers),
                "average_correlation": f"{metrics['avg_correlation']:.2f}",
                "concentration_risk": "High" if max(weights.values()) > 0.5 else "Moderate" if max(weights.values()) > 0.3 else "Low"
            },
            "risk_assessment": {
                "risk_level": "HIGH" if metrics['annual_volatility'] > 0.25 else "MODERATE" if metrics['annual_volatility'] > 0.15 else "LOW",
                "market_sensitivity": "High" if metrics['beta'] and metrics['beta'] > 1.2 else "Moderate" if metrics['beta'] and metrics['beta'] > 0.8 else "Low" if metrics['beta'] else "Unknown"
            },
            "data_period": f"{metrics['data_points']} trading days"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_stock_correlation(ticker1: str, ticker2: str) -> str:
    """
    Calculate correlation between two stocks.
    
    Args:
        ticker1: First stock ticker symbol
        ticker2: Second stock ticker symbol
    
    Returns:
        Correlation coefficient and interpretation
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = yf.download(
            [ticker1, ticker2], 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=True
        )
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        
        returns = data.pct_change().dropna()
        
        correlation = returns.corr().iloc[0, 1]
        
        if correlation > 0.7:
            interpretation = "strongly positively correlated"
        elif correlation > 0.3:
            interpretation = "moderately positively correlated"
        elif correlation > -0.3:
            interpretation = "weakly correlated"
        elif correlation > -0.7:
            interpretation = "moderately negatively correlated"
        else:
            interpretation = "strongly negatively correlated"
        
        return f"Correlation between {ticker1} and {ticker2}: {correlation:.3f} ({interpretation})"
        
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"


@tool
def compare_portfolio_to_benchmark(portfolio_input: str, benchmark: str = "SPY") -> str:
    """
    Compare portfolio performance to a benchmark index.
    
    Args:
        portfolio_input: Portfolio specification in format "TICKER1:WEIGHT1,TICKER2:WEIGHT2"
        benchmark: Benchmark ticker (default: SPY for S&P 500)
    
    Returns:
        Comparison metrics
    """
    try:
        tickers, weights = parse_portfolio_input(portfolio_input)
        
        if not tickers:
            return "No valid ticker symbols found"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Get portfolio data
        portfolio_data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=True
        )
        
        # Handle multi-level columns
        if isinstance(portfolio_data.columns, pd.MultiIndex):
            portfolio_data = portfolio_data['Close']
        
        portfolio_returns = portfolio_data.pct_change().dropna()
        weights_array = np.array([weights[t] for t in portfolio_data.columns])
        portfolio_total_returns = (portfolio_returns * weights_array).sum(axis=1)
        
        # Get benchmark data
        benchmark_data = yf.download(
            benchmark, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=True
        )
        
        if isinstance(benchmark_data, pd.DataFrame):
            if 'Close' in benchmark_data.columns:
                benchmark_data = benchmark_data['Close']
            else:
                benchmark_data = benchmark_data.iloc[:, 0]
        
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        # Calculate metrics
        portfolio_cum_return = (1 + portfolio_total_returns).prod() - 1
        benchmark_cum_return = (1 + benchmark_returns).prod() - 1
        
        portfolio_volatility = portfolio_total_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        outperformance = portfolio_cum_return - benchmark_cum_return
        
        result = f"""
Portfolio vs {benchmark} Comparison:
- Portfolio Return: {portfolio_cum_return*100:.2f}%
- {benchmark} Return: {benchmark_cum_return*100:.2f}%
- Outperformance: {outperformance*100:.2f}%
- Portfolio Volatility: {portfolio_volatility*100:.2f}%
- {benchmark} Volatility: {benchmark_volatility*100:.2f}%
"""
        return result
        
    except Exception as e:
        return f"Error comparing to benchmark: {str(e)}"

class PortfolioAnalyzer:
    def __init__(self, lookback_days=365):
        self.lookback_days = lookback_days
        self.risk_free_rate = 0.04
        
    def fetch_data(self, tickers: List[str]) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        try:
            # Download with auto_adjust=True to fix the warning
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True
            )
            
            # Handle different data structures
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    # Multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        data = data['Close']
                    else:
                        data = data[['Close']]
                elif len(data.columns) == len(tickers):
                    # Already just close prices
                    pass
                else:
                    # Single ticker case
                    data = pd.DataFrame(data['Close'])
                    data.columns = tickers
            
            # Convert to DataFrame if Series
            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = tickers
            
            # Drop any rows with NaN values
            data = data.dropna()
            
            if data.empty:
                raise ValueError("No data available after cleaning")
                
            return data
            
        except Exception as e:
            raise Exception(f"Failed to fetch data: {str(e)}")
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().dropna()
    
    def calculate_portfolio_metrics(self, tickers: List[str], weights: Dict[str, float]) -> Dict:
        try:
            prices = self.fetch_data(tickers)
        except Exception as e:
            return {"error": f"Failed to fetch data: {str(e)}"}
        
        if prices.empty:
            return {"error": "No price data available"}
        
        returns = self.calculate_returns(prices)
        weights_array = np.array([weights.get(ticker, 0) for ticker in prices.columns])
        portfolio_returns = (returns * weights_array).sum(axis=1)
        
        # Volatility
        daily_volatility = portfolio_returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Returns
        cumulative_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + cumulative_return) ** (252 / len(portfolio_returns)) - 1
        
        # Sharpe Ratio
        excess_return = annual_return - self.risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility != 0 else 0
        
        # Beta
        try:
            spy_data = yf.download('SPY', start=prices.index[0], end=prices.index[-1], progress=False)['Adj Close']
            spy_returns = spy_data.pct_change().dropna()
            
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            portfolio_aligned = portfolio_returns.loc[common_dates]
            spy_aligned = spy_returns.loc[common_dates]
            
            covariance = np.cov(portfolio_aligned, spy_aligned)[0][1]
            market_variance = np.var(spy_aligned)
            beta = covariance / market_variance if market_variance != 0 else 1.0
        except:
            beta = None
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Correlation
        correlation_matrix = returns.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "beta": beta,
            "max_drawdown": max_drawdown,
            "avg_correlation": avg_correlation,
            "data_points": len(portfolio_returns)
        }


def parse_portfolio_input(portfolio_input: str) -> tuple[List[str], Dict[str, float]]:
    """Parse portfolio input string into tickers and weights"""
    
    # Format 1: "AAPL:0.4,MSFT:0.3,GOOGL:0.3"
    if ":" in portfolio_input:
        parts = portfolio_input.split(",")
        tickers = []
        weights = {}
        
        for part in parts:
            ticker, weight = part.strip().split(":")
            ticker = ticker.strip().upper()
            tickers.append(ticker)
            weights[ticker] = float(weight)
        
        return tickers, weights
    
    # Format 2: "AAPL,MSFT,GOOGL" (equal weights)
    else:
        tickers = [t.strip().upper() for t in portfolio_input.split(",")]
        equal_weight = 1.0 / len(tickers)
        weights = {t: equal_weight for t in tickers}
        
        return tickers, weights

# System prompt for the portfolio agent
SYSTEM_PROMPT = """You are a Portfolio Insight Agent - an expert financial analyst specializing in portfolio analysis and risk management.

Your capabilities:
1. Analyze stock portfolios and calculate risk-return metrics
2. Explain portfolio characteristics in clear, conversational language
3. Identify strengths and weaknesses in portfolio allocation
4. Provide actionable insights about diversification and risk

You have access to these tools:
- analyze_portfolio: Calculate comprehensive portfolio metrics (volatility, Sharpe ratio, beta, max drawdown, etc.)
- get_stock_correlation: Check correlation between two stocks
- compare_portfolio_to_benchmark: Compare portfolio performance vs S&P 500

When a user asks about their portfolio:
1. Parse their input to identify tickers and weights (if not specified, assume equal weights)
2. Use the analyze_portfolio tool with format "TICKER1:WEIGHT1,TICKER2:WEIGHT2" (e.g., "AAPL:0.4,MSFT:0.6")
3. Interpret the results in a clear, conversational manner
4. Highlight key insights, strengths, and concerns
5. Provide context for the numbers (what does a Sharpe ratio of 1.5 mean?)

Be conversational and helpful. Use emojis sparingly for visual appeal. Focus on actionable insights."""


def create_agent_node(state: PortfolioState) -> PortfolioState:
    """Main agent node that uses LLM with tools"""
    
    # Bind tools to LLM
    tools = [analyze_portfolio, get_stock_correlation, compare_portfolio_to_benchmark]
    llm_with_tools = llm.bind_tools(tools)
    
    # Prepare messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # Get LLM response
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def tool_node(state: PortfolioState) -> PortfolioState:
    """Execute tools based on tool calls in the last message"""
    last_message = state["messages"][-1]
    
    # Get tool calls from the last message
    tool_calls = last_message.tool_calls
    
    # Map tool names to actual tool functions
    tools_map = {
        "analyze_portfolio": analyze_portfolio,
        "get_stock_correlation": get_stock_correlation,
        "compare_portfolio_to_benchmark": compare_portfolio_to_benchmark
    }
    
    # Execute each tool call
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Get the tool function
        tool_func = tools_map.get(tool_name)
        
        if tool_func:
            # Execute the tool
            try:
                result = tool_func.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found",
                    tool_call_id=tool_id,
                    name=tool_name
                )
            )
    
    return {"messages": tool_messages}


def should_continue(state: PortfolioState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end"""
    last_message = state["messages"][-1]
    
    # If LLM makes a tool call, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise end
    return "end"


def create_portfolio_agent():
    """Create the portfolio insight agent with LLM"""
    
    workflow = StateGraph(PortfolioState)
    
    # Add nodes
    workflow.add_node("agent", create_agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, return to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

def run_portfolio_agent(query: str):
    """Run the portfolio agent with a query"""
    
    agent = create_portfolio_agent()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "portfolio_data": {},
        "analysis_result": ""
    }
    
    result = agent.invoke(initial_state)
    
    return result["messages"][-1].content


if __name__ == "__main__":
    print(" Portfolio Insight Agent (LLM-Powered)")
    print(f"   Using: {LLM_PROVIDER.upper()}")
    if LLM_PROVIDER == "groq":
        print(f"   Model: llama-3.3-70b-versatile")
    print("="*60)
    
    # Example queries
    test_queries = [
        "Analyze my portfolio of AAPL, MSFT, and GOOGL with equal weights",
        "I have 50% in TSLA and 50% in NVDA. What's the risk level?",
        "How does a portfolio of 40% SPY, 30% QQQ, and 30% DIA perform?",
        "What's the correlation between AAPL and MSFT?",
    ]
    
    for query in test_queries:
        print(f"\n User: {query}\n")
        
        try:
            response = run_portfolio_agent(query)
            print(f" Agent: {response}")
        except Exception as e:
            print(f" Error: {str(e)}")
        
        print("\n" + "="*60)
    
    # Interactive mode
    print("\n Interactive Mode (type 'quit' to exit):\n")
    
    while True:
        user_input = input(" You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = run_portfolio_agent(user_input)
            print(f"\n Agent: {response}\n")
        except Exception as e:
            print(f"\n Error: {str(e)}\n")