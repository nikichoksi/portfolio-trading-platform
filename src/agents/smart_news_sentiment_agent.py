"""
Smart News Sentiment Agent
Fetches and analyzes news articles for stock tickers using web scraping.
"""

import os

# Set environment variables BEFORE any imports to prevent torchvision issues
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import requests
import re
import json
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import time
from dotenv import load_dotenv

# LLM imports for intelligent news source discovery
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic not available. Install with: pip install anthropic")

# Financial data imports
try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Install with: pip install yfinance pandas")

# Sentiment analysis imports
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

try:
    # Ensure env var is set before attempting import
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError, Exception) as e:
    TRANSFORMERS_AVAILABLE = False
    logging.warning(f"Transformers not available: {type(e).__name__}. Will use NLTK for sentiment analysis.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsSentimentAgent:
    """
    Agent for fetching and analyzing news sentiment for stock tickers.
    """
    
    def __init__(self):
        """
        Initialize the News Sentiment Agent.
        Sets up logging and API/base URLs.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing NewsSentimentAgent")
        
        # Base URLs for news sources
        self.yahoo_finance_base = "https://finance.yahoo.com/quote"
        self.finviz_base = "https://finviz.com/quote.ashx"
        self.yahoo_news_base = "https://finance.yahoo.com/quote/{}/news"
        
        # User agent to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Initialize LLM clients for news source discovery
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_llm_clients()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        self.use_huggingface = False
        self._initialize_sentiment_analyzer()
        
        self.logger.info("NewsSentimentAgent initialized successfully")
    
    def _initialize_llm_clients(self):
        """Initialize LLM clients for intelligent news source discovery"""
        # Try OpenAI first
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = openai.OpenAI(api_key=api_key)
                    self.logger.info("OpenAI client initialized for news source discovery")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Try Anthropic as fallback
        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    self.anthropic_client = Anthropic(api_key=api_key)
                    self.logger.info("Anthropic client initialized for news source discovery")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Anthropic client: {str(e)}")
        
        self.logger.info("No LLM client available. Will use default news sources only.")
    
    def _initialize_sentiment_analyzer(self):
        """
        Initialize sentiment analyzer (VADER or Hugging Face).
        Tries Hugging Face first, falls back to VADER.
        """
        # Try Hugging Face first (more accurate but heavier)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.logger.info("Initializing Hugging Face sentiment analyzer...")
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # Use CPU (-1) or GPU (0+)
                )
                self.use_huggingface = True
                self.logger.info("Hugging Face sentiment analyzer initialized")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize Hugging Face analyzer: {str(e)}")
                self.logger.info("Falling back to VADER sentiment analyzer")
        
        # Fall back to VADER (lighter, good for financial news)
        if NLTK_AVAILABLE:
            try:
                # Download VADER lexicon if not already downloaded
                try:
                    nltk.data.find('vader_lexicon')
                except LookupError:
                    self.logger.info("Downloading VADER lexicon...")
                    nltk.download('vader_lexicon', quiet=True)
                
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.use_huggingface = False
                self.logger.info("VADER sentiment analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize VADER analyzer: {str(e)}")
                self.sentiment_analyzer = None
        else:
            self.logger.error("Neither NLTK nor Transformers available. Sentiment analysis disabled.")
            self.sentiment_analyzer = None
    
    def fetch_news(self, ticker: str, max_articles: int = 10) -> List[Dict[str, str]]:
        """
        Fetch the latest news headlines and summaries for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            max_articles: Maximum number of articles to fetch (default: 10)
        
        Returns:
            List of dictionaries containing news articles with keys:
            - title: Article headline
            - summary: Article summary/description
            - link: URL to the full article
        """
        ticker = ticker.upper().strip()
        self.logger.info(f"Fetching news for ticker: {ticker}")
        
        articles = []
        
        try:
            # Step 1: Use LLM to find company-specific news sources
            llm_news_sources = self._find_company_news_sources(ticker)
            
            # Step 2: Try LLM-discovered news sources first (if available)
            if llm_news_sources:
                self.logger.info(f"Using {len(llm_news_sources)} LLM-discovered news sources for {ticker}")
                for source_url in llm_news_sources:
                    try:
                        # Check if it's a Yahoo Finance or Finviz URL (use existing methods)
                        if 'finance.yahoo.com' in source_url and f'/{ticker}/' in source_url:
                            yahoo_articles = self._fetch_yahoo_finance_news(ticker, max_articles)
                            articles.extend(yahoo_articles)
                        elif 'finviz.com' in source_url:
                            finviz_articles = self._fetch_finviz_news(ticker, max_articles - len(articles))
                            articles.extend(finviz_articles)
                        else:
                            # Try to scrape the URL directly
                            custom_articles = self._fetch_custom_news_source(source_url, ticker, max_articles - len(articles))
                            articles.extend(custom_articles)
                    except Exception as e:
                        self.logger.warning(f"Error fetching from LLM-discovered source {source_url}: {str(e)}")
                        continue
                    
                    # If we have enough articles, break
                    if len(articles) >= max_articles:
                        break
            
            # Step 3: Fallback to default sources if LLM didn't find enough or wasn't available
            if len(articles) < max_articles:
                self.logger.info(f"Using default news sources for {ticker} (found {len(articles)} articles so far)")
                # Try Yahoo Finance first
                yahoo_articles = self._fetch_yahoo_finance_news(ticker, max_articles)
                articles.extend(yahoo_articles)
                
                # If we don't have enough articles, try Finviz
                if len(articles) < max_articles:
                    finviz_articles = self._fetch_finviz_news(ticker, max_articles - len(articles))
                    articles.extend(finviz_articles)
            
            # Remove duplicates based on title similarity
            articles = self._remove_duplicates(articles)
            
            # Filter to prioritize ticker-specific articles
            # Check if articles mention the ticker symbol or company name
            ticker_upper = ticker.upper()
            
            # Common company name variations (basic mapping - can be expanded)
            company_keywords = {
                'AAPL': ['apple', 'iphone', 'ipad', 'macbook', 'tim cook'],
                'MSFT': ['microsoft', 'azure', 'office', 'windows', 'satya nadella'],
                'GOOGL': ['google', 'alphabet', 'android', 'youtube', 'sundar pichai'],
                'GOOG': ['google', 'alphabet', 'android', 'youtube'],
                'AMZN': ['amazon', 'aws', 'jeff bezos', 'alexa', 'prime'],
                'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'mark zuckerberg'],
                'TSLA': ['tesla', 'elon musk', 'model s', 'model 3', 'cybertruck'],
                'NVDA': ['nvidia', 'gpu', 'jensen huang', 'ai chip'],
            }
            
            # Get relevant keywords for this ticker
            relevant_keywords = [ticker_upper, ticker_upper.lower()]
            if ticker_upper in company_keywords:
                relevant_keywords.extend(company_keywords[ticker_upper])
            
            # Score articles based on relevance
            scored_articles = []
            for article in articles:
                title = article.get('title', '').upper()
                summary = article.get('summary', '').upper()
                text = f"{title} {summary}"
                
                # Calculate relevance score
                score = 0
                ticker_mentioned = False
                
                # Higher score if ticker is mentioned
                if ticker_upper in text:
                    score += 10
                    ticker_mentioned = True
                
                # Check for company-related keywords
                for keyword in relevant_keywords:
                    if keyword.upper() in text:
                        score += 5
                        break
                
                # Lower score for general market news keywords
                general_market_keywords = [
                    'DOW', 'S&P', 'NASDAQ', 'MARKET TODAY', 'STOCK MARKET',
                    'WHITE HOUSE', 'TRUMP', 'BIDEN', 'POLITICS', 'ELECTION'
                ]
                for keyword in general_market_keywords:
                    if keyword in text and not ticker_mentioned:
                        score -= 3  # Penalize general market news if ticker not mentioned
                
                article['_relevance_score'] = score
                article['_ticker_mentioned'] = ticker_mentioned
                scored_articles.append(article)
            
            # Sort by relevance score (highest first)
            scored_articles.sort(key=lambda x: x.get('_relevance_score', 0), reverse=True)
            
            # Prioritize ticker-specific articles
            ticker_mentioned_articles = [a for a in scored_articles if a.get('_ticker_mentioned', False)]
            other_articles = [a for a in scored_articles if not a.get('_ticker_mentioned', False)]
            
            # If we have enough ticker-specific articles, use only those
            if len(ticker_mentioned_articles) >= max_articles:
                articles = ticker_mentioned_articles[:max_articles]
                self.logger.info(f"Found {len(articles)} ticker-specific articles for {ticker}")
            elif len(ticker_mentioned_articles) > 0:
                # Use ticker-specific first, then top-scored related news
                # But prefer ticker-specific heavily
                articles = ticker_mentioned_articles + other_articles[:max_articles - len(ticker_mentioned_articles)]
                self.logger.info(f"Found {len(ticker_mentioned_articles)} ticker-specific and {len(articles) - len(ticker_mentioned_articles)} related articles for {ticker}")
            else:
                # No ticker-specific articles found - use top-scored articles
                articles = scored_articles[:max_articles]
                self.logger.warning(f"No ticker-specific articles found for {ticker}. Using top {len(articles)} articles.")
            
            # Remove internal scoring fields
            articles = [
                {k: v for k, v in article.items() if not k.startswith('_')}
                for article in articles
            ]
            
            # Limit to max_articles
            articles = articles[:max_articles]
            
            self.logger.info(f"Successfully fetched {len(articles)} articles for {ticker}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {ticker}: {str(e)}", exc_info=True)
            return []
    
    def _fetch_yahoo_finance_news(self, ticker: str, max_articles: int) -> List[Dict[str, str]]:
        """
        Fetch news from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles to fetch
        
        Returns:
            List of news article dictionaries
        """
        articles = []
        
        try:
            # Yahoo Finance news URL
            url = f"{self.yahoo_finance_base}/{ticker}/news"
            self.logger.debug(f"Fetching from Yahoo Finance: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles - Yahoo Finance structure
            # Look for article containers
            news_items = soup.find_all('li', class_='js-stream-content') or \
                        soup.find_all('div', {'data-module': 'NewsStream'}) or \
                        soup.find_all('h3', class_='Mb(5px)')
            
            # Alternative: Look for links with news
            if not news_items:
                # Try finding links in the news section
                news_section = soup.find('div', {'id': 'quoteNewsStream-0-Stream'}) or \
                              soup.find('div', class_='js-stream-content')
                
                if news_section:
                    news_items = news_section.find_all('a', href=True)
            
            # Parse articles
            for item in news_items[:max_articles * 2]:  # Get more to filter
                try:
                    article = {}
                    
                    # Find title
                    title_elem = item.find('h3') or item.find('a') or item
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        if title:
                            # Filter out advertisement/promoted content
                            title_lower = title.lower()
                            
                            # Skip if it's clearly an ad or unrelated content
                            skip_keywords = [
                                'advertisement', 'ad', 'sponsored', 'promoted',
                                'subscribe', 'sign up', 'click here', 'learn more'
                            ]
                            
                            if any(keyword in title_lower for keyword in skip_keywords):
                                continue
                            
                            # Skip if title is too short or looks like navigation
                            if len(title) < 10 or title.lower() in ['more', 'news', 'read', 'view']:
                                continue
                            
                            article['title'] = title
                            
                            # Find link
                            link_elem = item.find('a', href=True) or title_elem if title_elem.name == 'a' else None
                            if link_elem and link_elem.get('href'):
                                link = link_elem['href']
                                if link.startswith('/'):
                                    link = f"https://finance.yahoo.com{link}"
                                
                                # Skip if link looks like an ad or navigation
                                link_lower = link.lower()
                                if any(skip in link_lower for skip in ['/ad/', '/advertising/', '/promo/', '/subscribe']):
                                    continue
                                
                                article['link'] = link
                            else:
                                article['link'] = url
                            
                            # Find summary
                            summary_elem = item.find('p') or item.find('span', class_='C(#959595)')
                            if summary_elem:
                                summary = summary_elem.get_text(strip=True)
                                if summary and len(summary) > 20:  # Filter out very short summaries
                                    article['summary'] = summary
                                else:
                                    article['summary'] = title  # Use title as summary if no summary found
                            else:
                                article['summary'] = title
                            
                            # Check if article is relevant to the ticker
                            # Priority: articles that mention the ticker in title or summary
                            title_text = f"{article['title']} {article.get('summary', '')}".upper()
                            ticker_mentioned = ticker in title_text
                            
                            if article.get('title'):
                                article_data = {
                                    'title': article['title'],
                                    'summary': article.get('summary', article['title']),
                                    'link': article.get('link', url),
                                    '_ticker_mentioned': ticker_mentioned  # Internal flag for sorting
                                }
                                articles.append(article_data)
                
                except Exception as e:
                    self.logger.debug(f"Error parsing article: {str(e)}")
                    continue
            
            # Sort articles: prioritize those mentioning the ticker
            articles.sort(key=lambda x: (not x.get('_ticker_mentioned', False), x['title']))
            
            # Remove the internal flag and limit to max_articles
            articles = [
                {k: v for k, v in article.items() if k != '_ticker_mentioned'}
                for article in articles[:max_articles]
            ]
            
            # If no articles found with standard methods, try alternative approach
            if not articles:
                # Try searching for news stream items
                stream_items = soup.find_all('div', class_='Ov(h)')
                for item in stream_items[:max_articles]:
                    try:
                        link_elem = item.find('a', href=True)
                        if link_elem:
                            title = link_elem.get_text(strip=True)
                            if title:
                                link = link_elem.get('href', '')
                                if link.startswith('/'):
                                    link = f"https://finance.yahoo.com{link}"
                                
                                # Try to get summary from parent or sibling
                                parent = item.find_parent()
                                summary_elem = parent.find('p') if parent else None
                                summary = summary_elem.get_text(strip=True) if summary_elem else title
                                
                                articles.append({
                                    'title': title,
                                    'summary': summary[:200] if len(summary) > 200 else summary,
                                    'link': link
                                })
                    except Exception as e:
                        self.logger.debug(f"Error parsing stream item: {str(e)}")
                        continue
            
        except requests.RequestException as e:
            self.logger.error(f"Network error fetching Yahoo Finance news: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error parsing Yahoo Finance news: {str(e)}", exc_info=True)
        
        return articles
    
    def _fetch_finviz_news(self, ticker: str, max_articles: int) -> List[Dict[str, str]]:
        """
        Fetch news from Finviz.
        
        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles to fetch
        
        Returns:
            List of news article dictionaries
        """
        articles = []
        
        try:
            # Finviz news URL
            url = f"{self.finviz_base}?t={ticker}"
            self.logger.debug(f"Fetching from Finviz: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news table in Finviz
            news_table = soup.find('table', {'id': 'news-table'})
            
            if news_table:
                # Find all news rows
                news_rows = news_table.find_all('tr', class_='cursor-pointer') or \
                           news_table.find_all('tr')
                
                for row in news_rows[:max_articles]:
                    try:
                        # Find link and title
                        link_elem = row.find('a', href=True, class_='tab-link-news')
                        if link_elem:
                            title = link_elem.get_text(strip=True)
                            link = link_elem.get('href', '')
                            
                            if title:
                                articles.append({
                                    'title': title,
                                    'summary': title,  # Finviz usually doesn't have summaries
                                    'link': link if link.startswith('http') else f"https://finviz.com{link}"
                                })
                    except Exception as e:
                        self.logger.debug(f"Error parsing Finviz row: {str(e)}")
                        continue
        
        except requests.RequestException as e:
            self.logger.error(f"Network error fetching Finviz news: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error parsing Finviz news: {str(e)}", exc_info=True)
        
        return articles
    
    def _remove_duplicates(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove duplicate articles based on title similarity.
        
        Args:
            articles: List of article dictionaries
        
        Returns:
            List of unique articles
        """
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _find_company_news_sources(self, ticker: str, company_name: Optional[str] = None) -> List[str]:
        """
        Use LLM to intelligently find company-specific news pages and sources.
        
        This method uses an LLM to discover the best news sources for a company,
        avoiding hardcoding and working for any ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., "GOOGL")
            company_name: Optional company name (e.g., "Google")
        
        Returns:
            List of news source URLs prioritized for the company
        """
        if not self.openai_client and not self.anthropic_client:
            self.logger.debug("No LLM client available. Using default news sources.")
            return []
        
        try:
            # Get company information using yfinance if available
            company_info = None
            if YFINANCE_AVAILABLE:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    company_name = info.get('longName') or info.get('shortName') or company_name
                    company_info = info.get('longBusinessSummary', '')
                except Exception as e:
                    self.logger.debug(f"Could not fetch company info from yfinance: {str(e)}")
            
            # Determine company name if not provided
            if not company_name:
                # Common ticker to company name mapping (fallback)
                ticker_to_company = {
                    'AAPL': 'Apple Inc',
                    'MSFT': 'Microsoft Corporation',
                    'GOOGL': 'Google (Alphabet Inc)',
                    'GOOG': 'Google (Alphabet Inc)',
                    'AMZN': 'Amazon.com Inc',
                    'META': 'Meta Platforms Inc (Facebook)',
                    'NVDA': 'Nvidia Corporation',
                    'TSLA': 'Tesla Inc',
                }
                company_name = ticker_to_company.get(ticker.upper(), f"Company with ticker {ticker}")
            
            # Create prompt for LLM to find company-specific news sources
            company_search_term = company_name.replace(' ', '+')
            prompt = f"""Find the best news sources and URLs specifically for {company_name} (stock ticker: {ticker}).

Requirements:
1. Provide company-specific news page URLs (e.g., Yahoo Finance ticker page, Finviz quote page)
2. Focus on financial news sites with dedicated sections for {ticker}
3. Include company press release pages if available
4. Avoid general market news pages

Please return a JSON array of URLs. Examples:
- Yahoo Finance: https://finance.yahoo.com/quote/{ticker}/news
- Finviz: https://finviz.com/quote.ashx?t={ticker}
- Google News search: https://news.google.com/search?q={company_search_term}+stock

Return ONLY a valid JSON array of strings, no other text. Maximum 5 URLs."""

            # Call LLM
            if self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that finds company-specific financial news sources. Always return a valid JSON array of URLs."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=300
                    )
                    result = response.choices[0].message.content.strip()
                except Exception as e:
                    self.logger.warning(f"Error calling OpenAI: {str(e)}")
                    return []
            elif self.anthropic_client:
                try:
                    response = self.anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=300,
                        temperature=0.2,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    result = response.content[0].text.strip()
                except Exception as e:
                    self.logger.warning(f"Error calling Anthropic: {str(e)}")
                    return []
            else:
                return []
            
            # Parse JSON response
            # Remove markdown code blocks if present
            result = re.sub(r'```json\s*', '', result)
            result = re.sub(r'```\s*', '', result)
            result = result.strip()
            
            # Try to parse as JSON
            try:
                # Handle both array and object responses
                parsed = json.loads(result)
                
                # If it's a dict, look for common keys
                if isinstance(parsed, dict):
                    # Common keys LLMs might use
                    for key in ['urls', 'sources', 'news_sources', 'links', 'results']:
                        if key in parsed and isinstance(parsed[key], list):
                            news_sources = parsed[key]
                            break
                    else:
                        # Try to find any list in the dict
                        news_sources = [v for v in parsed.values() if isinstance(v, list)]
                        if news_sources:
                            news_sources = news_sources[0]
                        else:
                            news_sources = []
                elif isinstance(parsed, list):
                    news_sources = parsed
                else:
                    news_sources = []
                
                # Validate and filter URLs
                valid_urls = []
                for url in news_sources:
                    if isinstance(url, str) and url.startswith(('http://', 'https://')):
                        valid_urls.append(url)
                    elif isinstance(url, dict):
                        # If URL is in a dict, try common keys
                        for key in ['url', 'link', 'source', 'href']:
                            if key in url and isinstance(url[key], str) and url[key].startswith(('http://', 'https://')):
                                valid_urls.append(url[key])
                                break
                
                if valid_urls:
                    self.logger.info(f"Found {len(valid_urls)} news sources for {ticker} using LLM")
                    return valid_urls[:5]  # Limit to top 5 sources
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract URLs from text
                urls = re.findall(r'https?://[^\s\)"<>]+', result)
                valid_urls = [url for url in urls if any(domain in url for domain in ['yahoo.com', 'finviz.com', 'google.com', 'bloomberg.com', 'reuters.com', 'cnbc.com', 'wsj.com', 'marketwatch.com'])]
                if valid_urls:
                    self.logger.info(f"Extracted {len(valid_urls)} URLs for {ticker} using LLM")
                    return valid_urls[:5]
            
        except Exception as e:
            self.logger.warning(f"Error finding news sources with LLM for {ticker}: {str(e)}")
            # Don't fail completely, just log warning and continue with default sources
        
        return []
    
    def _fetch_custom_news_source(self, url: str, ticker: str, max_articles: int) -> List[Dict[str, str]]:
        """
        Fetch news from a custom news source URL.
        
        Args:
            url: URL to scrape
            ticker: Stock ticker symbol (for filtering)
            max_articles: Maximum number of articles to fetch
        
        Returns:
            List of news article dictionaries
        """
        articles = []
        
        try:
            self.logger.debug(f"Fetching from custom news source: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find news articles - generic approach
            # Look for common news article patterns
            news_items = (
                soup.find_all('article') or
                soup.find_all('div', class_=re.compile(r'news|article|headline', re.I)) or
                soup.find_all('h3') or
                soup.find_all('h2') or
                soup.find_all('a', href=re.compile(r'news|article', re.I))
            )
            
            # Filter for articles that might be about the ticker
            ticker_upper = ticker.upper()
            for item in news_items[:max_articles * 2]:
                try:
                    # Find title
                    title_elem = item.find('a') or item.find('h3') or item.find('h2') or item
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        
                        if not title or len(title) < 10:
                            continue
                        
                        # Filter out ads and navigation
                        title_lower = title.lower()
                        skip_keywords = [
                            'advertisement', 'ad', 'sponsored', 'promoted',
                            'subscribe', 'sign up', 'click here', 'learn more'
                        ]
                        
                        if any(keyword in title_lower for keyword in skip_keywords):
                            continue
                        
                        # Find link
                        link_elem = item.find('a', href=True) or (title_elem if title_elem.name == 'a' else None)
                        link = None
                        if link_elem and link_elem.get('href'):
                            link = link_elem['href']
                            if link.startswith('/'):
                                # Make absolute URL
                                link = urljoin(url, link)
                        else:
                            link = url
                        
                        # Find summary
                        summary_elem = item.find('p') or item.find('span', class_=re.compile(r'summary|description|excerpt', re.I))
                        summary = summary_elem.get_text(strip=True) if summary_elem else title
                        
                        # Only include if it mentions the ticker or seems relevant
                        text = f"{title} {summary}".upper()
                        if ticker_upper in text or len(title) > 30:
                            articles.append({
                                'title': title,
                                'summary': summary[:200] if len(summary) > 200 else summary,
                                'link': link
                            })
                            
                            if len(articles) >= max_articles:
                                break
                
                except Exception as e:
                    self.logger.debug(f"Error parsing custom news item: {str(e)}")
                    continue
        
        except requests.RequestException as e:
            self.logger.warning(f"Network error fetching custom news source {url}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error parsing custom news source {url}: {str(e)}")
        
        return articles
    
    def analyze_sentiment(self, news_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles.
        
        Args:
            news_data: List of news article dictionaries with 'title' and optionally 'summary'
        
        Returns:
            Dictionary containing:
            - avg_sentiment: Average sentiment score (-1 to +1)
            - sentiment_breakdown: Dictionary with sentiment counts and article details
            - articles: List of articles with sentiment analysis
        """
        if not news_data:
            self.logger.warning("Empty news data provided for sentiment analysis")
            return {
                "avg_sentiment": 0.0,
                "sentiment_breakdown": {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "total": 0
                },
                "articles": []
            }
        
        if self.sentiment_analyzer is None:
            self.logger.error("Sentiment analyzer not initialized. Cannot analyze sentiment.")
            return {
                "avg_sentiment": 0.0,
                "sentiment_breakdown": {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "total": 0,
                    "error": "Sentiment analyzer not available"
                },
                "articles": []
            }
        
        self.logger.info(f"Analyzing sentiment for {len(news_data)} articles")
        
        analyzed_articles = []
        sentiment_scores = []
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for article in news_data:
            try:
                # Get text to analyze (prefer summary, fall back to title)
                text = article.get('summary', article.get('title', ''))
                
                if not text:
                    self.logger.warning(f"Skipping article with no text: {article}")
                    continue
                
                # Analyze sentiment
                if self.use_huggingface:
                    sentiment_result = self._analyze_with_huggingface(text)
                else:
                    sentiment_result = self._analyze_with_vader(text)
                
                # Convert to numeric score
                sentiment_label = sentiment_result["sentiment"]
                sentiment_score = sentiment_result["score"]
                
                # Map to numeric value: positive = +1, neutral = 0, negative = -1
                if sentiment_label == "positive":
                    numeric_score = sentiment_score  # 0.0 to 1.0 -> 0.0 to 1.0
                    sentiment_counts["positive"] += 1
                elif sentiment_label == "negative":
                    numeric_score = -sentiment_score  # 0.0 to 1.0 -> -1.0 to 0.0
                    sentiment_counts["negative"] += 1
                else:  # neutral
                    numeric_score = 0.0
                    sentiment_counts["neutral"] += 1
                
                sentiment_scores.append(numeric_score)
                
                # Add sentiment to article
                analyzed_article = {
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "link": article.get("link", ""),
                    "sentiment": sentiment_label,
                    "score": round(sentiment_score, 3)
                }
                analyzed_articles.append(analyzed_article)
                
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment for article: {str(e)}", exc_info=True)
                continue
        
        # Calculate average sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            avg_sentiment = round(avg_sentiment, 3)
        else:
            avg_sentiment = 0.0
        
        # Build sentiment breakdown
        total_articles = len(analyzed_articles)
        sentiment_breakdown = {
            "positive": sentiment_counts["positive"],
            "neutral": sentiment_counts["neutral"],
            "negative": sentiment_counts["negative"],
            "total": total_articles,
            "positive_percentage": round((sentiment_counts["positive"] / total_articles * 100), 2) if total_articles > 0 else 0,
            "neutral_percentage": round((sentiment_counts["neutral"] / total_articles * 100), 2) if total_articles > 0 else 0,
            "negative_percentage": round((sentiment_counts["negative"] / total_articles * 100), 2) if total_articles > 0 else 0
        }
        
        self.logger.info(f"Sentiment analysis complete. Avg sentiment: {avg_sentiment}")
        
        return {
            "avg_sentiment": avg_sentiment,
            "sentiment_breakdown": sentiment_breakdown,
            "articles": analyzed_articles
        }
    
    def _analyze_with_huggingface(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using Hugging Face transformers.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with 'sentiment' and 'score'
        """
        try:
            # Hugging Face returns [{'label': 'POSITIVE' or 'NEGATIVE', 'score': 0.0-1.0}]
            result = self.sentiment_analyzer(text)[0]
            
            label = result['label'].upper()
            score = result['score']
            
            # Hugging Face model only returns POSITIVE or NEGATIVE
            # Consider neutral if score is close to 0.5
            if score < 0.55 and score > 0.45:
                sentiment = "neutral"
                score = 0.5
            elif label == "POSITIVE":
                sentiment = "positive"
            else:
                sentiment = "negative"
            
            return {
                "sentiment": sentiment,
                "score": round(score, 3)
            }
        except Exception as e:
            self.logger.error(f"Error in Hugging Face sentiment analysis: {str(e)}")
            return {"sentiment": "neutral", "score": 0.5}
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using VADER sentiment analyzer.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with 'sentiment' and 'score'
        """
        try:
            # VADER returns {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': -1.0 to 1.0}
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            compound = scores['compound']
            
            # Determine sentiment based on compound score
            # compound >= 0.05: positive
            # compound <= -0.05: negative
            # else: neutral
            if compound >= 0.05:
                sentiment = "positive"
                # Score is the confidence: use the positive score or compound score
                score = max(scores['pos'], abs(compound))
            elif compound <= -0.05:
                sentiment = "negative"
                # Score is the confidence: use the negative score or absolute compound score
                score = max(scores['neg'], abs(compound))
            else:
                sentiment = "neutral"
                # Score reflects how neutral (closer to 0 = more neutral)
                score = scores['neu']
            
            return {
                "sentiment": sentiment,
                "score": round(score, 3)
            }
        except Exception as e:
            self.logger.error(f"Error in VADER sentiment analysis: {str(e)}")
            return {"sentiment": "neutral", "score": 0.5}
    
    def detect_event_type(self, news_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Detect event types from news articles using keyword/phrase detection.
        
        Args:
            news_data: List of news article dictionaries with 'title' and optionally 'summary'
        
        Returns:
            List of dictionaries containing:
            - title: Article title
            - event: Detected event type
            - confidence: Confidence score (0.0 to 1.0)
            - keywords_matched: List of keywords that matched
        """
        if not news_data:
            self.logger.warning("Empty news data provided for event detection")
            return []
        
        # Event type keyword mappings
        event_keywords = {
            "product_launch": [
                "launch", "launches", "launched", "releases", "released", "release",
                "introduces", "introduced", "introduction", "unveils", "unveiled",
                "unveiling", "debuts", "debut", "debuted", "announces", "announced",
                "announcement", "rolls out", "rolled out", "rollout", "preview",
                "previews", "previewed", "unveiling", "unveil", "coming soon",
                "available now", "now available"
            ],
            "acquisition": [
                "acquires", "acquired", "acquisition", "merger", "merged", "merges",
                "merge", "takeover", "took over", "taking over", "purchased company",
                "buyout", "bought out", "acquiring", "consolidation", "consolidated",
                "consolidates", "bought company", "acquired company", "merger deal",
                "takeover bid", "merges with", "acquires stake", "acquisition deal"
            ],
            "earnings_report": [
                "earnings", "revenue", "revenues", "quarter", "quarters", "q1", "q2",
                "q3", "q4", "fiscal", "financial results", "financial report",
                "quarterly results", "annual results", "quarterly earnings",
                "annual earnings", "profit", "profits", "profitability", "loss",
                "losses", "beat earnings", "beats earnings", "missed earnings",
                "miss earnings", "forecast", "forecasts", "guidance", "eps",
                "earnings per share", "sales report", "outperformed", "underperformed"
            ],
            "partnership": [
                "partnership", "partnerships", "partner", "partners", "partnered",
                "partnering", "collaboration", "collaborates", "collaborated",
                "collaborating", "collaborative", "alliance", "alliances",
                "joins forces", "teaming up", "team up", "teamed up", "work together",
                "working together", "joint venture", "joint ventures", "strategic",
                "agreement", "agreements", "deal", "deals"
            ],
            "negative_event": [
                "lawsuit", "lawsuits", "sued", "suing", "legal action", "legal actions",
                "delay", "delays", "delayed", "postponed", "postponement",
                "recall", "recalls", "recalled", "recalling", "recalled",
                "investigation", "investigations", "investigating", "investigated",
                "violation", "violations", "fined", "fine", "fines", "penalty",
                "penalties", "breach", "breaches", "scandal", "scandals",
                "crisis", "crises", "controversy", "controversies", "allegations",
                "alleged", "charges", "charged", "probe", "probes", "probed",
                "failure", "failures", "failed", "failing", "downgrade", "downgrades",
                "downgraded", "cut", "cuts", "reduction", "reductions", "layoff",
                "layoffs", "fired", "firing", "terminated", "termination"
            ]
        }
        
        self.logger.info(f"Detecting event types for {len(news_data)} articles")
        
        detected_events = []
        event_counts = {}
        
        for article in news_data:
            try:
                # Get text to analyze (prefer summary, fall back to title)
                title = article.get('title', '')
                summary = article.get('summary', '')
                text = f"{title} {summary}".lower()
                
                if not text.strip():
                    self.logger.warning(f"Skipping article with no text: {article}")
                    continue
                
                # Detect events in the text
                detected_event_types = []
                matched_keywords = []
                
                for event_type, keywords in event_keywords.items():
                    matches = []
                    for keyword in keywords:
                        # Check if keyword appears in text (word boundary matching)
                        keyword_lower = keyword.lower()
                        # More precise matching: check if it's a whole word
                        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                        if re.search(pattern, text):
                            matches.append(keyword)
                    
                    if matches:
                        # Calculate confidence based on number of matches
                        # Base confidence increases with more keyword matches
                        base_confidence = min(0.5 + (len(matches) - 1) * 0.15, 0.9)
                        confidence = base_confidence
                        
                        # Boost confidence for certain important keywords
                        important_keywords = {
                            "product_launch": ["launch", "releases", "unveils", "announces"],
                            "acquisition": ["acquires", "merger", "buys", "takeover"],
                            "earnings_report": ["earnings", "revenue", "quarter"],
                            "partnership": ["partnership", "collaboration", "alliance"],
                            "negative_event": ["lawsuit", "recall", "investigation", "violation"]
                        }
                        
                        if any(kw in matches for kw in important_keywords.get(event_type, [])):
                            confidence = min(confidence + 0.2, 1.0)
                        
                        detected_event_types.append({
                            "event": event_type,
                            "confidence": round(confidence, 3),
                            "keywords_matched": matches
                        })
                        
                        matched_keywords.extend(matches)
                
                # Rank events by confidence and importance
                if detected_event_types:
                    # Sort by confidence (highest first)
                    detected_event_types.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Select the primary event (highest confidence)
                    primary_event = detected_event_types[0]
                    
                    # If multiple events with similar confidence, prioritize by importance
                    if len(detected_event_types) > 1:
                        # Importance order: earnings_report > acquisition > negative_event > product_launch > partnership
                        importance_order = {
                            "earnings_report": 5,
                            "acquisition": 4,
                            "negative_event": 3,
                            "product_launch": 2,
                            "partnership": 1
                        }
                        
                        # Check if there's a more important event with similar confidence
                        for event in detected_event_types[1:]:
                            if (event['confidence'] >= primary_event['confidence'] - 0.2 and
                                importance_order.get(event['event'], 0) > importance_order.get(primary_event['event'], 0)):
                                primary_event = event
                    
                    # Add to detected events
                    detected_events.append({
                        "title": title,
                        "event": primary_event['event'],
                        "confidence": primary_event['confidence'],
                        "keywords_matched": list(set(primary_event['keywords_matched']))
                    })
                    
                    # Update event counts
                    event_type = primary_event['event']
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                else:
                    # No event detected
                    detected_events.append({
                        "title": title,
                        "event": "general_news",
                        "confidence": 0.0,
                        "keywords_matched": []
                    })
                    event_counts["general_news"] = event_counts.get("general_news", 0) + 1
                
            except Exception as e:
                self.logger.error(f"Error detecting event type for article: {str(e)}", exc_info=True)
                continue
        
        # Log event detection summary
        self.logger.info(f"Event detection complete. Events found: {event_counts}")
        
        return detected_events
    
    def analyze_historical_growth(self, ticker: str, event_type: str) -> Dict[str, Any]:
        """
        Analyze historical stock growth after similar events.
        
        Fetches historical stock data for the past 3 years and simulates finding
        past occurrences of similar events, then calculates average price changes
        after those events.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            event_type: Type of event to analyze (e.g., "product_launch", "acquisition")
        
        Returns:
            Dictionary containing:
            - event: Event type
            - avg_growth_7d: Average % change in 7 days after event
            - avg_growth_30d: Average % change in 30 days after event
            - num_events_found: Number of similar events found
            - error: Error message if any
        """
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance is not available. Cannot analyze historical growth.")
            return {
                "event": event_type,
                "avg_growth_7d": None,
                "avg_growth_30d": None,
                "num_events_found": 0,
                "error": "yfinance library not available"
            }
        
        ticker = ticker.upper().strip()
        self.logger.info(f"Analyzing historical growth for {ticker} after {event_type} events")
        
        try:
            # Fetch historical stock data for the past 3 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3 * 365)
            
            self.logger.info(f"Fetching historical data from {start_date.date()} to {end_date.date()}")
            
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                self.logger.warning(f"No historical data found for {ticker}")
                return {
                    "event": event_type,
                    "avg_growth_7d": None,
                    "avg_growth_30d": None,
                    "num_events_found": 0,
                    "error": f"No historical data found for {ticker}"
                }
            
            # Ensure we have a 'Close' column
            if 'Close' not in hist_data.columns:
                self.logger.error(f"No 'Close' price data found for {ticker}")
                return {
                    "event": event_type,
                    "avg_growth_7d": None,
                    "avg_growth_30d": None,
                    "num_events_found": 0,
                    "error": "No close price data available"
                }
            
            # Simulate finding past occurrences of similar events
            # This is a placeholder - in a real implementation, you would:
            # 1. Fetch historical news articles for the ticker
            # 2. Use detect_event_type on those articles
            # 3. Filter by the event_type
            # 4. Get the dates of those events
            
            # For now, we'll use a heuristic approach:
            # - Find dates with significant price movements (potential event dates)
            # - Or use a placeholder dataset with simulated event dates
            
            event_dates = self._simulate_event_dates(hist_data, event_type)
            
            if not event_dates:
                self.logger.warning(f"No simulated events found for {event_type} for {ticker}")
                return {
                    "event": event_type,
                    "avg_growth_7d": None,
                    "avg_growth_30d": None,
                    "num_events_found": 0,
                    "error": f"No events found for {event_type}"
                }
            
            self.logger.info(f"Found {len(event_dates)} simulated events of type {event_type}")
            
            # Calculate price changes after each event
            growth_7d_list = []
            growth_30d_list = []
            
            for event_date in event_dates:
                try:
                    # Convert event_date to Timestamp if needed
                    event_date_ts = pd.Timestamp(event_date)
                    
                    # Find the closest date in hist_data to the event_date
                    # (since event_date might not be exactly in the index due to weekends/holidays)
                    if event_date_ts not in hist_data.index:
                        # Find the closest date in the index
                        closest_index = hist_data.index.get_indexer([event_date_ts], method='nearest')[0]
                        if closest_index == -1:
                            self.logger.warning(f"Could not find closest date for {event_date}")
                            continue
                        event_date_ts = hist_data.index[closest_index]
                    
                    # Get the price on the event date
                    event_price = hist_data.loc[event_date_ts, 'Close']
                    
                    # Calculate 7-day growth
                    date_7d = event_date_ts + timedelta(days=7)
                    # Find closest date to 7 days later
                    future_dates_7d = hist_data.index[hist_data.index > event_date_ts]
                    if len(future_dates_7d) > 0:
                        # Find date closest to 7 days
                        target_date_7d = event_date_ts + timedelta(days=7)
                        closest_date_7d = min(future_dates_7d, key=lambda x: abs((x - target_date_7d).days))
                        if abs((closest_date_7d - target_date_7d).days) <= 3:  # Within 3 days of 7
                            price_7d = hist_data.loc[closest_date_7d, 'Close']
                            growth_7d = ((price_7d - event_price) / event_price) * 100
                            growth_7d_list.append(growth_7d)
                    
                    # Calculate 30-day growth
                    date_30d = event_date_ts + timedelta(days=30)
                    # Find closest date to 30 days later
                    future_dates_30d = hist_data.index[hist_data.index > event_date_ts]
                    if len(future_dates_30d) > 0:
                        # Find date closest to 30 days
                        target_date_30d = event_date_ts + timedelta(days=30)
                        closest_date_30d = min(future_dates_30d, key=lambda x: abs((x - target_date_30d).days))
                        if abs((closest_date_30d - target_date_30d).days) <= 5:  # Within 5 days of 30
                            price_30d = hist_data.loc[closest_date_30d, 'Close']
                            growth_30d = ((price_30d - event_price) / event_price) * 100
                            growth_30d_list.append(growth_30d)
                
                except (KeyError, IndexError) as e:
                    self.logger.warning(f"Error processing event date {event_date}: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error processing event date {event_date}: {str(e)}")
                    continue
            
            # Calculate average growth
            avg_growth_7d = None
            avg_growth_30d = None
            
            if growth_7d_list:
                avg_growth_7d = sum(growth_7d_list) / len(growth_7d_list)
                avg_growth_7d = round(avg_growth_7d, 2)
            
            if growth_30d_list:
                avg_growth_30d = sum(growth_30d_list) / len(growth_30d_list)
                avg_growth_30d = round(avg_growth_30d, 2)
            
            result = {
                "event": event_type,
                "avg_growth_7d": avg_growth_7d,
                "avg_growth_30d": avg_growth_30d,
                "num_events_found": len(event_dates)
            }
            
            self.logger.info(f"Historical growth analysis complete: {result}")
            return result
        
        except Exception as e:
            self.logger.error(f"Error analyzing historical growth for {ticker}: {str(e)}", exc_info=True)
            return {
                "event": event_type,
                "avg_growth_7d": None,
                "avg_growth_30d": None,
                "num_events_found": 0,
                "error": f"Error analyzing historical growth: {str(e)}"
            }
    
    def _simulate_event_dates(self, hist_data: pd.DataFrame, event_type: str) -> List[datetime]:
        """
        Simulate finding past event dates based on historical data.
        
        This is a placeholder method. In a real implementation, you would:
        1. Fetch historical news articles
        2. Use detect_event_type on those articles
        3. Return the dates of articles matching the event_type
        
        For now, we'll use a heuristic:
        - Find dates with significant price movements (volatility spikes)
        - Or use quarterly dates for earnings_report events
        - Or use random dates as a fallback
        
        Args:
            hist_data: Historical stock price data
            event_type: Type of event to simulate
        
        Returns:
            List of datetime objects representing event dates
        """
        event_dates = []
        
        try:
            if hist_data.empty or 'Close' not in hist_data.columns:
                return event_dates
            
            # Convert index to datetime if it's not already
            if not isinstance(hist_data.index, pd.DatetimeIndex):
                hist_data.index = pd.to_datetime(hist_data.index)
            
            # Strategy 1: For earnings_report, use quarterly dates
            if event_type == "earnings_report":
                # Find dates that are likely earnings dates (end of quarters)
                # Typically: Jan 31, Apr 30, Jul 31, Oct 31 (or close to these)
                quarterly_dates = []
                for date in hist_data.index:
                    month = date.month
                    day = date.day
                    # Check if it's near end of quarter (last 10 days of Jan, Apr, Jul, Oct)
                    if month in [1, 4, 7, 10] and day >= 21:
                        quarterly_dates.append(date)
                
                # Limit to max 12 events (3 years * 4 quarters)
                if quarterly_dates:
                    event_dates = sorted(quarterly_dates)[-12:]
                else:
                    # Fallback: use dates with significant price movements
                    hist_data_copy = hist_data.copy()
                    hist_data_copy['Returns'] = hist_data_copy['Close'].pct_change()
                    hist_data_copy['AbsReturns'] = hist_data_copy['Returns'].abs()
                    if len(hist_data_copy) > 10:
                        threshold = hist_data_copy['AbsReturns'].quantile(0.90)
                        volatile_dates = hist_data_copy[hist_data_copy['AbsReturns'] >= threshold].index
                        event_dates = sorted(volatile_dates)[-12:] if len(volatile_dates) > 0 else []
                    else:
                        event_dates = []
            
            # Strategy 2: For other events, find dates with significant price movements
            else:
                # Calculate daily returns
                hist_data_copy = hist_data.copy()
                hist_data_copy['Returns'] = hist_data_copy['Close'].pct_change()
                hist_data_copy['AbsReturns'] = hist_data_copy['Returns'].abs()
                
                # Find dates with high volatility (top 10% of absolute returns)
                if len(hist_data_copy) > 10:  # Need at least 10 data points
                    threshold = hist_data_copy['AbsReturns'].quantile(0.90)
                    volatile_dates = hist_data_copy[hist_data_copy['AbsReturns'] >= threshold].index
                    
                    # Limit to reasonable number of events (max 10-15)
                    if len(volatile_dates) > 0:
                        event_dates = sorted(volatile_dates)[-12:]
                    else:
                        event_dates = []
                else:
                    event_dates = []
            
            # Convert to list of datetime objects (already Timestamps from pandas)
            if event_dates:
                event_dates = [pd.Timestamp(date).to_pydatetime() for date in event_dates]
                
                # Remove duplicates and sort
                event_dates = sorted(list(set(event_dates)))
                
                # Ensure dates are within the historical data range and have enough future data
                if hist_data.index.min() and hist_data.index.max():
                    min_date = pd.Timestamp(hist_data.index.min())
                    max_date = pd.Timestamp(hist_data.index.max())
                    # Only keep dates that are at least 30 days before the end
                    cutoff_date = max_date - timedelta(days=30)
                    event_dates = [
                        date for date in event_dates
                        if min_date <= pd.Timestamp(date) <= cutoff_date
                    ]
            
            self.logger.info(f"Simulated {len(event_dates)} event dates for {event_type}")
            
        except Exception as e:
            self.logger.error(f"Error simulating event dates: {str(e)}", exc_info=True)
            # Fallback: return empty list
            return []
        
        return event_dates
    
    def simulate_risk_profit(self, avg_sentiment: float, avg_growth_7d: float) -> Dict[str, Any]:
        """
        Simulate risk and profit potential based on sentiment and historical growth.
        
        Combines sentiment analysis and historical growth data into a weighted score
        to assess risk level and profit potential. Handles negative sentiment and
        historical losses correctly.
        
        Args:
            avg_sentiment: Average sentiment score (-1 to +1)
                - Positive values indicate positive sentiment
                - Negative values indicate negative sentiment
            avg_growth_7d: Average 7-day growth percentage (can be positive or negative)
                - Positive values indicate historical gains
                - Negative values indicate historical losses
        
        Returns:
            Dictionary containing:
            - risk_level: Risk level category ("Low", "Medium", "High", "Very High")
            - profit_potential: Profit potential score (0-100)
            - score: Combined weighted score (-1 to +1)
            - sentiment_contribution: Contribution from sentiment (weighted)
            - growth_contribution: Contribution from historical growth (weighted)
        """
        try:
            # Validate inputs
            if avg_sentiment is None:
                avg_sentiment = 0.0
            if avg_growth_7d is None:
                avg_growth_7d = 0.0
            
            # Ensure sentiment is in range [-1, +1]
            avg_sentiment = max(-1.0, min(1.0, float(avg_sentiment)))
            
            # Normalize avg_growth_7d to range [-1, +1]
            # Divide by 10 to normalize (assuming typical range is -10% to +10%)
            # For extreme values, cap at -1 and +1
            normalized_growth = max(-1.0, min(1.0, float(avg_growth_7d) / 10.0))
            
            # Calculate weighted score
            # Weight: sentiment 60%, historical growth 40%
            sentiment_weight = 0.6
            growth_weight = 0.4
            
            sentiment_contribution = avg_sentiment * sentiment_weight
            growth_contribution = normalized_growth * growth_weight
            score = sentiment_contribution + growth_contribution
            
            # Ensure score is in range [-1, +1]
            score = max(-1.0, min(1.0, score))
            
            # Determine risk level and profit potential based on score
            if score > 0.5:
                risk_level = "Low"
                # Profit potential: 80-100 (higher score = higher profit)
                # Map score from (0.5, 1.0] to [80, 100]
                # When score = 0.5 -> profit = 80, when score = 1.0 -> profit = 100
                profit_potential = 80 + ((score - 0.5) / 0.5) * 20
            elif score > 0:
                risk_level = "Medium"
                # Profit potential: 50-70
                # Map score from (0, 0.5] to [50, 70]
                # When score = 0 -> profit = 50, when score = 0.5 -> profit = 70
                profit_potential = 50 + (score / 0.5) * 20
            elif score >= -0.5:
                risk_level = "High"
                # Profit potential: 20-40
                # Map score from [-0.5, 0] to [20, 40]
                # When score = -0.5 -> profit = 20, when score = 0 -> profit = 40
                # Linear interpolation: profit = 20 + ((score - (-0.5)) / (0 - (-0.5))) * (40 - 20)
                profit_potential = 20 + ((score + 0.5) / 0.5) * 20
            else:  # score < -0.5
                risk_level = "Very High"
                # Profit potential: 0-20
                # Map score from [-1.0, -0.5) to [0, 20]
                # When score = -1.0 -> profit = 0, when score = -0.5 -> profit = 20
                profit_potential = 0 + ((score + 1.0) / (-0.5 + 1.0)) * 20
            
            # Ensure profit_potential is in range [0, 100]
            profit_potential = max(0, min(100, round(profit_potential)))
            
            self.logger.info(
                f"Risk/Profit simulation: sentiment={avg_sentiment:.3f}, "
                f"growth_7d={avg_growth_7d:.2f}%, score={score:.3f}, "
                f"risk={risk_level}, profit={profit_potential}"
            )
            
            return {
                "risk_level": risk_level,
                "profit_potential": int(profit_potential),
                "score": round(score, 3),
                "sentiment_contribution": round(sentiment_contribution, 3),
                "growth_contribution": round(growth_contribution, 3),
                "avg_sentiment": round(avg_sentiment, 3),
                "avg_growth_7d": round(avg_growth_7d, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating risk/profit: {str(e)}", exc_info=True)
            # Return default values on error
            return {
                "risk_level": "High",
                "profit_potential": 30,
                "score": 0.0,
                "sentiment_contribution": 0.0,
                "growth_contribution": 0.0,
                "avg_sentiment": avg_sentiment if avg_sentiment is not None else 0.0,
                "avg_growth_7d": avg_growth_7d if avg_growth_7d is not None else 0.0,
                "error": str(e)
            }
    
    def run(self, ticker: str, max_articles: int = 10, save_to_json: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete news sentiment and risk analysis pipeline for a ticker.
        
        Orchestrates the following pipeline:
        1. fetch_news() - Fetch latest news articles
        2. analyze_sentiment() - Analyze sentiment of news articles
        3. detect_event_type() - Detect event types from news
        4. analyze_historical_growth() - Analyze historical growth after similar events
        5. simulate_risk_profit() - Simulate risk and profit potential
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            max_articles: Maximum number of articles to fetch (default: 10)
            save_to_json: Whether to save results to JSON file (default: False)
            output_dir: Directory to save JSON file (default: "data" directory)
        
        Returns:
            Dictionary containing:
            - ticker: Ticker symbol
            - avg_sentiment: Average sentiment score (-1 to +1)
            - event: Most common event type detected
            - avg_growth_7d: Average 7-day growth after similar events
            - avg_growth_30d: Average 30-day growth after similar events
            - risk_level: Risk level ("Low", "Medium", "High", "Very High")
            - profit_potential: Profit potential score (0-100)
            - latest_headlines: List of latest news headlines
            - insight: Generated insight based on sentiment and historical data
            - sentiment_breakdown: Sentiment breakdown statistics
            - event_breakdown: Event type breakdown
            - error: Error message if any
            - saved_to: Path to saved JSON file (if save_to_json=True)
        """
        ticker = ticker.upper().strip()
        self.logger.info(f"Running complete pipeline for {ticker}")
        
        result = {
            "ticker": ticker,
            "avg_sentiment": None,
            "event": None,
            "avg_growth_7d": None,
            "avg_growth_30d": None,
            "risk_level": None,
            "profit_potential": None,
            "latest_headlines": [],
            "insight": None,
            "sentiment_breakdown": None,
            "event_breakdown": None,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Fetch news
            self.logger.info(f"Step 1: Fetching news for {ticker}")
            articles = self.fetch_news(ticker, max_articles=max_articles)
            
            if not articles:
                result["error"] = f"No news articles found for {ticker}"
                result["insight"] = f"No recent news found for {ticker}. Unable to perform sentiment analysis."
                self.logger.warning(f"No articles found for {ticker}")
                return result
            
            # Extract latest headlines - TICKER-CENTRIC ONLY
            # Filter to show only articles that mention the ticker or company name
            ticker_upper = ticker.upper()
            ticker_headlines = []
            
            # Company keyword mapping for better filtering
            company_keywords = {
                'AAPL': ['APPLE', 'IPHONE', 'IPAD', 'MACBOOK', 'TIM COOK'],
                'MSFT': ['MICROSOFT', 'AZURE', 'OFFICE', 'WINDOWS', 'SATYA NADELLA'],
                'GOOGL': ['GOOGLE', 'ALPHABET', 'ANDROID', 'YOUTUBE', 'SUNDAR PICHAI'],
                'GOOG': ['GOOGLE', 'ALPHABET', 'ANDROID', 'YOUTUBE'],
                'AMZN': ['AMAZON', 'AWS', 'ALEXA', 'PRIME', 'JEFF BEZOS'],
                'META': ['META', 'FACEBOOK', 'INSTAGRAM', 'WHATSAPP', 'MARK ZUCKERBERG'],
                'TSLA': ['TESLA', 'ELON MUSK', 'MODEL S', 'MODEL 3', 'CYBERTRUCK'],
                'NVDA': ['NVIDIA', 'GPU', 'AI CHIP', 'JENSEN HUANG'],
            }
            
            # Get relevant keywords for this ticker
            relevant_keywords = [ticker_upper]
            if ticker_upper in company_keywords:
                relevant_keywords.extend(company_keywords[ticker_upper])
            
            # Filter articles to only those mentioning the ticker or company
            for article in articles:
                title = article.get("title", "")
                summary = article.get("summary", "")
                text = f"{title} {summary}".upper()
                
                # Check if article mentions ticker or company keywords
                is_ticker_related = False
                
                # Check for ticker symbol
                if ticker_upper in text:
                    is_ticker_related = True
                # Check for company keywords
                elif ticker_upper in company_keywords:
                    if any(keyword in text for keyword in company_keywords[ticker_upper]):
                        is_ticker_related = True
                
                # Only include ticker-related articles in headlines
                if is_ticker_related:
                    ticker_headlines.append({
                        "title": title,
                        "link": article.get("link", "N/A")
                    })
                    
                    # Limit to top 5 ticker-specific headlines
                    if len(ticker_headlines) >= 5:
                        break
            
            # If we don't have enough ticker-specific headlines, use top-scored articles
            # But log a warning that we're showing non-ticker-specific news
            if len(ticker_headlines) < 5:
                remaining = [a for a in articles if a.get("title") not in [h["title"] for h in ticker_headlines]]
                if remaining:
                    self.logger.warning(
                        f"Only found {len(ticker_headlines)} ticker-specific headlines for {ticker}. "
                        f"Adding top {min(5 - len(ticker_headlines), len(remaining))} articles."
                    )
                    for article in remaining[:5 - len(ticker_headlines)]:
                        ticker_headlines.append({
                            "title": article.get("title", "N/A"),
                            "link": article.get("link", "N/A")
                        })
            
            result["latest_headlines"] = ticker_headlines[:5]
            
            # Log what we're showing
            if len(ticker_headlines) > 0:
                ticker_specific_count = sum(
                    1 for h in ticker_headlines 
                    if ticker_upper in f"{h['title']}".upper() or 
                    (ticker_upper in company_keywords and 
                     any(kw in f"{h['title']}".upper() for kw in company_keywords[ticker_upper]))
                )
                self.logger.info(
                    f"Showing {len(ticker_headlines)} headlines for {ticker}: "
                    f"{ticker_specific_count} ticker-specific, {len(ticker_headlines) - ticker_specific_count} related"
                )
            
            # Step 2: Analyze sentiment
            # NOTE: We analyze ALL fetched articles (including general market news) for sentiment
            # This provides better context for sentiment analysis
            # But "latest_headlines" shown to user are ticker-centric only
            self.logger.info(f"Step 2: Analyzing sentiment for {ticker} using {len(articles)} articles")
            sentiment_result = self.analyze_sentiment(articles)
            
            avg_sentiment = sentiment_result.get("avg_sentiment", 0.0)
            result["avg_sentiment"] = avg_sentiment
            result["sentiment_breakdown"] = sentiment_result.get("sentiment_breakdown", {})
            
            # Step 3: Detect event types
            self.logger.info(f"Step 3: Detecting event types for {ticker}")
            event_result = self.detect_event_type(articles)
            
            if not event_result:
                result["error"] = "No events detected"
                result["insight"] = f"Sentiment analysis for {ticker} shows average sentiment of {avg_sentiment:.2f}, but no specific events were detected."
                return result
            
            # Count events by type
            event_counts = {}
            for event in event_result:
                event_type = event.get("event", "unknown")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            result["event_breakdown"] = event_counts
            
            # Get most common event type (excluding general_news)
            most_common_event = None
            if event_counts:
                # Filter out general_news if there are other events
                filtered_events = {k: v for k, v in event_counts.items() if k != "general_news"}
                if filtered_events:
                    most_common_event = max(filtered_events.items(), key=lambda x: x[1])[0]
                else:
                    # If only general_news, use it
                    most_common_event = max(event_counts.items(), key=lambda x: x[1])[0]
            
            result["event"] = most_common_event or "general_news"
            
            # Step 4: Analyze historical growth
            if most_common_event and most_common_event != "general_news":
                self.logger.info(f"Step 4: Analyzing historical growth for {ticker} after {most_common_event} events")
                growth_result = self.analyze_historical_growth(ticker, most_common_event)
                
                result["avg_growth_7d"] = growth_result.get("avg_growth_7d")
                result["avg_growth_30d"] = growth_result.get("avg_growth_30d")
                
                # Step 5: Simulate risk/profit
                if result["avg_growth_7d"] is not None:
                    self.logger.info(f"Step 5: Simulating risk/profit for {ticker}")
                    risk_profit_result = self.simulate_risk_profit(
                        avg_sentiment,
                        result["avg_growth_7d"]
                    )
                    
                    result["risk_level"] = risk_profit_result.get("risk_level")
                    result["profit_potential"] = risk_profit_result.get("profit_potential")
                else:
                    # Use sentiment alone if no growth data
                    self.logger.warning(f"No growth data for {ticker}, using sentiment only")
                    # Normalize sentiment to approximate growth
                    estimated_growth = avg_sentiment * 5.0  # Rough estimate
                    risk_profit_result = self.simulate_risk_profit(
                        avg_sentiment,
                        estimated_growth
                    )
                    result["risk_level"] = risk_profit_result.get("risk_level")
                    result["profit_potential"] = risk_profit_result.get("profit_potential")
            else:
                # No specific event found, use sentiment only
                self.logger.info(f"No specific events found for {ticker}, using sentiment only")
                estimated_growth = avg_sentiment * 5.0  # Rough estimate based on sentiment
                risk_profit_result = self.simulate_risk_profit(
                    avg_sentiment,
                    estimated_growth
                )
                result["risk_level"] = risk_profit_result.get("risk_level")
                result["profit_potential"] = risk_profit_result.get("profit_potential")
            
            # Generate insight (pass news articles for context)
            result["insight"] = self._generate_insight(result, news_articles=articles)
            
            self.logger.info(f"Pipeline complete for {ticker}: risk={result['risk_level']}, profit={result['profit_potential']}")
            
        except Exception as e:
            self.logger.error(f"Error running pipeline for {ticker}: {str(e)}", exc_info=True)
            result["error"] = str(e)
            result["insight"] = f"Error analyzing {ticker}: {str(e)}"
        
        # Save to JSON if requested
        if save_to_json:
            try:
                import json
                from pathlib import Path
                
                # Determine output directory
                if output_dir is None:
                    # Default to data directory
                    output_dir = Path(__file__).parent.parent.parent / "data"
                else:
                    output_dir = Path(output_dir)
                
                # Create directory if it doesn't exist
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{ticker}_analysis_{timestamp}.json"
                output_file = output_dir / filename
                
                # Save to JSON
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, default=str, ensure_ascii=False)
                
                result["saved_to"] = str(output_file)
                self.logger.info(f"Results saved to: {output_file}")
                
            except Exception as e:
                self.logger.error(f"Error saving results to JSON: {str(e)}", exc_info=True)
                result["save_error"] = str(e)
        
        return result
    
    def _generate_insight(self, result: Dict[str, Any], news_articles: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate an insight string based on the analysis results using GPT API.
        Falls back to template-based insights if GPT is not available.
        
        Args:
            result: Dictionary containing analysis results
            news_articles: List of news articles (optional, for context)
        
        Returns:
            Insight string describing the findings
        """
        ticker = result.get("ticker", "Unknown")
        avg_sentiment = result.get("avg_sentiment", 0.0)
        event = result.get("event", "general_news")
        avg_growth_7d = result.get("avg_growth_7d")
        avg_growth_30d = result.get("avg_growth_30d")
        risk_level = result.get("risk_level", "Unknown")
        profit_potential = result.get("profit_potential", 0)
        sentiment_breakdown = result.get("sentiment_breakdown", {})
        event_breakdown = result.get("event_breakdown", {})
        
        # Try to generate insight using GPT API
        if self.openai_client:
            try:
                return self._generate_insight_with_gpt(
                    ticker=ticker,
                    avg_sentiment=avg_sentiment,
                    event=event,
                    avg_growth_7d=avg_growth_7d,
                    avg_growth_30d=avg_growth_30d,
                    risk_level=risk_level,
                    profit_potential=profit_potential,
                    sentiment_breakdown=sentiment_breakdown,
                    event_breakdown=event_breakdown,
                    news_articles=news_articles[:5] if news_articles else []
                )
            except Exception as e:
                self.logger.warning(f"Failed to generate insight with GPT: {e}. Falling back to template-based insight.")
        
        # Fallback to template-based insight
        return self._generate_insight_template(
            ticker=ticker,
            avg_sentiment=avg_sentiment,
            event=event,
            avg_growth_7d=avg_growth_7d,
            avg_growth_30d=avg_growth_30d,
            risk_level=risk_level,
            profit_potential=profit_potential
        )
    
    def _generate_insight_with_gpt(
        self,
        ticker: str,
        avg_sentiment: float,
        event: str,
        avg_growth_7d: Optional[float],
        avg_growth_30d: Optional[float],
        risk_level: str,
        profit_potential: int,
        sentiment_breakdown: Dict[str, Any],
        event_breakdown: Dict[str, Any],
        news_articles: List[Dict[str, str]]
    ) -> str:
        """
        Generate insight using OpenAI GPT API.
        
        Args:
            ticker: Stock ticker symbol
            avg_sentiment: Average sentiment score (-1 to +1)
            event: Most common event type
            avg_growth_7d: Average 7-day growth percentage
            avg_growth_30d: Average 30-day growth percentage
            risk_level: Risk level (Low, Medium, High, Very High)
            profit_potential: Profit potential score (0-100)
            sentiment_breakdown: Sentiment breakdown statistics
            event_breakdown: Event type breakdown
            news_articles: List of recent news articles
        
        Returns:
            Generated insight string
        """
        # Prepare news headlines for context
        news_context = ""
        if news_articles:
            headlines = [article.get("title", "") for article in news_articles[:5] if article.get("title")]
            if headlines:
                news_context = "\n".join([f"- {headline}" for headline in headlines])
        
        # Prepare sentiment breakdown
        sentiment_info = ""
        if sentiment_breakdown:
            positive = sentiment_breakdown.get("positive", 0)
            neutral = sentiment_breakdown.get("neutral", 0)
            negative = sentiment_breakdown.get("negative", 0)
            total = sentiment_breakdown.get("total", 0)
            if total > 0:
                sentiment_info = f"Sentiment breakdown: {positive} positive ({positive/total*100:.1f}%), {neutral} neutral ({neutral/total*100:.1f}%), {negative} negative ({negative/total*100:.1f}%) articles."
        
        # Prepare event breakdown
        event_info = ""
        if event_breakdown:
            events = [f"{k.replace('_', ' ').title()}: {v}" for k, v in sorted(event_breakdown.items(), key=lambda x: x[1], reverse=True)]
            if events:
                event_info = f"Detected events: {', '.join(events)}."
        
        # Prepare historical growth info
        growth_info = ""
        if avg_growth_7d is not None:
            growth_info += f"7-day average growth: {avg_growth_7d:.2f}%"
            if avg_growth_30d is not None:
                growth_info += f", 30-day average growth: {avg_growth_30d:.2f}%"
        
        # Build prompt for GPT
        analysis_lines = [
            f"Average Sentiment Score: {avg_sentiment:.2f} (range: -1 to +1, where >0.3 is positive, <-0.3 is negative)",
        ]
        
        if sentiment_info:
            analysis_lines.append(sentiment_info)
        
        analysis_lines.append(f"Most Common Event Type: {event.replace('_', ' ').title()}")
        
        if event_info:
            analysis_lines.append(event_info)
        
        if growth_info:
            analysis_lines.append(growth_info)
        else:
            analysis_lines.append("No historical growth data available")
        
        analysis_lines.extend([
            f"Risk Level: {risk_level}",
            f"Profit Potential: {profit_potential}% (0-100 scale)"
        ])
        
        analysis_data = "\n".join([f"- {line}" for line in analysis_lines])
        
        prompt = f"""You are a financial analyst providing investment insights. Based on the following analysis for {ticker} stock, generate a concise, actionable insight (2-3 sentences maximum).

Analysis Data:
{analysis_data}

Recent News Headlines:
{news_context if news_context else "No recent headlines available"}

Requirements:
1. Write in a clear, professional tone
2. Focus on actionable insights: what the news means and what investors should expect
3. If historical data is available, explain what it means for future performance
4. Mention the risk level and profit potential naturally
5. Keep it concise (2-3 sentences maximum)
6. Do NOT explain technical details or methods used
7. Focus on "what" and "so what" for the investor

Example format:
"According to recent news, {ticker} is experiencing [event type]. Historical data shows that similar events have led to [growth pattern]. Based on current sentiment and historical performance, this represents a [risk level] investment opportunity with [profit_potential]% profit potential."

Generate the insight now:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst providing concise, actionable investment insights. Focus on what matters to investors, not technical details."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            insight = response.choices[0].message.content.strip()
            self.logger.info(f"Generated GPT insight for {ticker}")
            return insight
            
        except Exception as e:
            self.logger.error(f"Error generating insight with GPT: {e}")
            raise
    
    def _generate_insight_template(
        self,
        ticker: str,
        avg_sentiment: float,
        event: str,
        avg_growth_7d: Optional[float],
        avg_growth_30d: Optional[float],
        risk_level: str,
        profit_potential: int
    ) -> str:
        """
        Generate template-based insight (fallback when GPT is not available).
        
        Args:
            ticker: Stock ticker symbol
            avg_sentiment: Average sentiment score (-1 to +1)
            event: Most common event type
            avg_growth_7d: Average 7-day growth percentage
            avg_growth_30d: Average 30-day growth percentage
            risk_level: Risk level (Low, Medium, High, Very High)
            profit_potential: Profit potential score (0-100)
        
        Returns:
            Template-based insight string
        """
        # Determine sentiment description
        if avg_sentiment > 0.3:
            sentiment_desc = "positive"
        elif avg_sentiment > -0.3:
            sentiment_desc = "neutral"
        else:
            sentiment_desc = "negative"
        
        # Build insight
        insight_parts = []
        
        # Sentiment
        insight_parts.append(f"{sentiment_desc.capitalize()} sentiment")
        
        # Event
        if event and event != "general_news":
            # Format event name for display
            event_display = " ".join(word.capitalize() for word in event.replace("_", " ").split())
            insight_parts.append(f"with {event_display.lower()} events detected")
        
        # Historical growth
        if avg_growth_7d is not None:
            if event and event != "general_news":
                # Format event name
                event_formatted = " ".join(word.capitalize() for word in event.replace("_", " ").split())
                if avg_growth_7d > 0:
                    insight_parts.append(f"shows average {avg_growth_7d:.1f}% gain 7 days after similar {event_formatted.lower()} events")
                else:
                    insight_parts.append(f"shows average {abs(avg_growth_7d):.1f}% decline 7 days after similar {event_formatted.lower()} events")
            else:
                if avg_growth_7d > 0:
                    insight_parts.append(f"with average {avg_growth_7d:.1f}% gain historically")
                else:
                    insight_parts.append(f"with average {abs(avg_growth_7d):.1f}% decline historically")
            
            if avg_growth_30d is not None:
                if avg_growth_30d > 0:
                    insight_parts.append(f"and {avg_growth_30d:.1f}% gain over 30 days")
                else:
                    insight_parts.append(f"and {abs(avg_growth_30d):.1f}% decline over 30 days")
        
        # Risk and profit
        insight_parts.append(f"indicating {risk_level.lower()} risk")
        insight_parts.append(f"with {profit_potential}% profit potential")
        
        insight = f"{' '.join(insight_parts)}."
        
        # Capitalize first letter
        if insight:
            insight = insight[0].upper() + insight[1:]
        
        return insight
    
    def __del__(self):
        """Cleanup: close session when agent is destroyed"""
        if hasattr(self, 'session'):
            self.session.close()


if __name__ == "__main__":
    """
    Demo for the NewsSentimentAgent.
    Runs the complete pipeline for AAPL as a demonstration.
    """
    import sys
    import json
    from pathlib import Path
    
    print("=" * 80)
    print("News Sentiment Agent - Full Pipeline Demo")
    print("=" * 80)
    
    # Create agent instance
    agent = NewsSentimentAgent()
    
    # Demo: Run full pipeline for AAPL
    demo_ticker = "AAPL"
    print(f"\nRunning complete pipeline for {demo_ticker}...")
    print("-" * 80)
    
    # Run pipeline and save to JSON
    demo_result = agent.run(demo_ticker, max_articles=10, save_to_json=True)
    
    print(f"\nPipeline Results for {demo_result['ticker']}:")
    print("-" * 80)
    print(f"Average Sentiment: {demo_result.get('avg_sentiment', 'N/A')}")
    print(f"Most Common Event: {demo_result.get('event', 'N/A')}")
    print(f"Average 7-Day Growth: {demo_result.get('avg_growth_7d', 'N/A')}%")
    print(f"Average 30-Day Growth: {demo_result.get('avg_growth_30d', 'N/A')}%")
    print(f"Risk Level: {demo_result.get('risk_level', 'N/A')}")
    print(f"Profit Potential: {demo_result.get('profit_potential', 'N/A')}")
    print(f"\nInsight: {demo_result.get('insight', 'N/A')}")
    
    if demo_result.get('sentiment_breakdown'):
        print(f"\nSentiment Breakdown:")
        breakdown = demo_result['sentiment_breakdown']
        print(f"  Positive: {breakdown.get('positive', 0)} ({breakdown.get('positive_percentage', 0)}%)")
        print(f"  Neutral: {breakdown.get('neutral', 0)} ({breakdown.get('neutral_percentage', 0)}%)")
        print(f"  Negative: {breakdown.get('negative', 0)} ({breakdown.get('negative_percentage', 0)}%)")
    
    if demo_result.get('event_breakdown'):
        print(f"\nEvent Breakdown:")
        for event_type, count in sorted(demo_result['event_breakdown'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {event_type}: {count}")
    
    if demo_result.get('latest_headlines'):
        print(f"\nLatest Headlines:")
        for i, headline in enumerate(demo_result['latest_headlines'][:5], 1):
            print(f"  {i}. {headline.get('title', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("Full JSON Result:")
    print("=" * 80)
    print(json.dumps(demo_result, indent=2, default=str))
    print("=" * 80)
    
    # Show where results were saved
    if demo_result.get('saved_to'):
        print(f"\nResults saved to: {demo_result['saved_to']}")
    elif demo_result.get('save_error'):
        print(f"\nWarning: Could not save results: {demo_result['save_error']}")
    print("=" * 80)

