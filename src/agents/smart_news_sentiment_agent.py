"""
Smart News Sentiment Agent
Fetches and analyzes news articles for stock tickers using web scraping.
"""

import os
import logging
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from datetime import datetime
import time
from dotenv import load_dotenv

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
        
        self.logger.info("NewsSentimentAgent initialized successfully")
    
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
            # Try Yahoo Finance first
            yahoo_articles = self._fetch_yahoo_finance_news(ticker, max_articles)
            articles.extend(yahoo_articles)
            
            # If we don't have enough articles, try Finviz
            if len(articles) < max_articles:
                finviz_articles = self._fetch_finviz_news(ticker, max_articles - len(articles))
                articles.extend(finviz_articles)
            
            # Remove duplicates based on title similarity
            articles = self._remove_duplicates(articles)
            
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
            for item in news_items[:max_articles]:
                try:
                    article = {}
                    
                    # Find title
                    title_elem = item.find('h3') or item.find('a') or item
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        if title:
                            article['title'] = title
                            
                            # Find link
                            link_elem = item.find('a', href=True) or title_elem if title_elem.name == 'a' else None
                            if link_elem and link_elem.get('href'):
                                link = link_elem['href']
                                if link.startswith('/'):
                                    link = f"https://finance.yahoo.com{link}"
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
                            
                            if article.get('title'):
                                # Return only required fields: title, summary, link
                                articles.append({
                                    'title': article['title'],
                                    'summary': article.get('summary', article['title']),
                                    'link': article.get('link', url)
                                })
                
                except Exception as e:
                    self.logger.debug(f"Error parsing article: {str(e)}")
                    continue
            
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
    
    def __del__(self):
        """Cleanup: close session when agent is destroyed"""
        if hasattr(self, 'session'):
            self.session.close()


if __name__ == "__main__":
    """
    Quick test for the NewsSentimentAgent.
    """
    print("=" * 60)
    print("Testing NewsSentimentAgent")
    print("=" * 60)
    
    # Create agent instance
    agent = NewsSentimentAgent()
    
    # Test with AAPL
    ticker = "AAPL"
    print(f"\nFetching news for {ticker}...")
    print("-" * 60)
    
    articles = agent.fetch_news(ticker, max_articles=10)
    
    if articles:
        print(f"\n✓ Successfully fetched {len(articles)} articles for {ticker}\n")
        
        for i, article in enumerate(articles, 1):
            print(f"Article {i}:")
            print(f"  Title: {article.get('title', 'N/A')}")
            print(f"  Summary: {article.get('summary', 'N/A')[:100]}...")
            print(f"  Link: {article.get('link', 'N/A')}")
            print()
    else:
        print(f"\n✗ No articles found for {ticker}")
        print("This might be due to:")
        print("  1. Network connectivity issues")
        print("  2. Website structure changes")
        print("  3. Rate limiting or blocking")
        print("  4. Ticker symbol not found")
    
    print("=" * 60)

