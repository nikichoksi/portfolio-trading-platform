#!/usr/bin/env python3
"""Setup Snowflake with sample data"""

from src.database.snowflake_config import SnowflakeConfig
from src.etl.data_loader import SnowflakeDataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize Snowflake and load sample data"""
    
    # Test connection
    logger.info("Testing Snowflake connection...")
    config = SnowflakeConfig()
    if not config.get_connection():
        logger.error("❌ Cannot connect to Snowflake")
        return
    
    # Initialize schema
    logger.info("Initializing schema...")
    conn = config.get_connection()
    cursor = conn.cursor()
    
    with open('src/database/snowflake_input_schema.sql', 'r') as f:
        sql = f.read()
        for statement in sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info("✅ Schema initialized")
    
    # Load sample data
    logger.info("Loading sample data...")
    loader = SnowflakeDataLoader()
    
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'JPM', 'JNJ', 'V'
    ]
    
    loader.load_all_data(sample_tickers, period='2y')
    
    logger.info("✅ Setup complete!")

if __name__ == "__main__":
    main()