"""
Smart News Sentiment Agent
Fetches and analyzes news articles for stock tickers using web scraping.
"""

import os
import logging
import requests
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup
from datetime import datetime
import time
from dotenv import load_dotenv

# Sentiment analysis imports
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

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
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        self.use_huggingface = False
        self._initialize_sentiment_analyzer()
        
        self.logger.info("NewsSentimentAgent initialized successfully")
    
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
        
        # Analyze sentiment
        print("Analyzing sentiment...")
        print("-" * 60)
        sentiment_result = agent.analyze_sentiment(articles)
        
        # Display results
        print(f"\nSentiment Analysis Results:")
        print(f"  Average Sentiment: {sentiment_result['avg_sentiment']:.3f}")
        print(f"  Sentiment Breakdown:")
        breakdown = sentiment_result['sentiment_breakdown']
        print(f"    Positive: {breakdown['positive']} ({breakdown.get('positive_percentage', 0)}%)")
        print(f"    Neutral: {breakdown['neutral']} ({breakdown.get('neutral_percentage', 0)}%)")
        print(f"    Negative: {breakdown['negative']} ({breakdown.get('negative_percentage', 0)}%)")
        print(f"    Total: {breakdown['total']}")
        
        print(f"\nArticles with Sentiment:")
        print("-" * 60)
        for i, article in enumerate(sentiment_result['articles'], 1):
            print(f"\nArticle {i}:")
            print(f"  Title: {article.get('title', 'N/A')}")
            print(f"  Sentiment: {article.get('sentiment', 'N/A').upper()}")
            print(f"  Score: {article.get('score', 'N/A')}")
            print(f"  Link: {article.get('link', 'N/A')}")
    else:
        print(f"\n✗ No articles found for {ticker}")
        print("This might be due to:")
        print("  1. Network connectivity issues")
        print("  2. Website structure changes")
        print("  3. Rate limiting or blocking")
        print("  4. Ticker symbol not found")
    
    print("=" * 60)

