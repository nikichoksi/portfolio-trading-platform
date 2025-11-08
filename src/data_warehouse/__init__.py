"""
Data warehouse module for Snowflake integration.
Provides historical market data storage and analytics.
"""

from .snowflake_connector import SnowflakeWarehouse

__all__ = ['SnowflakeWarehouse']
