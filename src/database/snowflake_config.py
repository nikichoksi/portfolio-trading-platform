import os
from dotenv import load_dotenv
import snowflake.connector
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SnowflakeConfig:
    """Secure Snowflake configuration"""
    
    def __init__(self):
        # Load .env file
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"✅ Loaded .env file from {env_path.absolute()}")
        
        self.account = os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = os.getenv('SNOWFLAKE_USER')
        self.password = os.getenv('SNOWFLAKE_PASSWORD')
        self.database = os.getenv('SNOWFLAKE_DATABASE', 'PORTFOLIO_TRADING')
        self.schema = os.getenv('SNOWFLAKE_SCHEMA', 'ANALYTICS')
        self.warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
        self.role = os.getenv('SNOWFLAKE_ROLE', 'TRAINING_ROLE')  # Changed from ACCOUNTADMIN
        
        # 🔧 FIX: Add insecure mode flag for educational accounts
        self.insecure_mode = os.getenv('SNOWFLAKE_INSECURE_MODE', 'true').lower() == 'true'
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate required credentials"""
        required = {
            'account': self.account,
            'user': self.user,
            'password': self.password
        }
        
        missing = [k for k, v in required.items() if not v]
        
        if missing:
            error_msg = f"""
❌ Missing Snowflake credentials: {', '.join(missing)}

Please set them in your .env file:
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
"""
            raise ValueError(error_msg)
        
        logger.info(f"✅ Snowflake config validated for user: {self.user}")
    
    def get_connection(self):
        """Get Snowflake connection with SSL workaround for edu accounts"""
        
        # Build connection parameters
        conn_params = {
            'account': self.account,
            'user': self.user,
            'password': self.password,
            'role': self.role,
        }
        
        # Add database/schema/warehouse if specified
        if self.database and self.database != '<none selected>':
            conn_params['database'] = self.database
        if self.schema and self.schema != '<none selected>':
            conn_params['schema'] = self.schema
        if self.warehouse and self.warehouse != '<none selected>':
            conn_params['warehouse'] = self.warehouse
        
        # 🔧 FIX: Add insecure mode for educational accounts
        if self.insecure_mode:
            conn_params['insecure_mode'] = True
            logger.warning("⚠️ Connecting in INSECURE MODE (SSL verification disabled)")
        
        try:
            logger.info(f"🔌 Connecting to Snowflake account: {self.account}")
            conn = snowflake.connector.connect(**conn_params)
            logger.info("✅ Connected to Snowflake successfully")
            return conn
            
        except snowflake.connector.errors.OperationalError as e:
            if "Certificate did not match" in str(e) or "hostname" in str(e):
                logger.error(f"""
❌ SSL Certificate Error!

Your account identifier format might be wrong.

Current: {self.account}

Try these formats in your .env file:

1. SNOWFLAKE_ACCOUNT=sfedu02-vob68402  (with dash)
2. SNOWFLAKE_ACCOUNT=vob68402.sfedu02  (reversed)
3. SNOWFLAKE_ACCOUNT=VOB68402  (uppercase, no prefix)

Or check your Snowflake login URL to find the correct format.
""")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            raise
    
    def get_sqlalchemy_engine(self):
        """Get SQLAlchemy engine"""
        from sqlalchemy import create_engine
        
        # Build connection string
        connection_string = (
            f"snowflake://{self.user}:{self.password}@{self.account}/"
        )
        
        if self.database and self.database != '<none selected>':
            connection_string += f"{self.database}/"
        
        if self.schema and self.schema != '<none selected>':
            connection_string += f"{self.schema}"
        
        connection_string += f"?warehouse={self.warehouse}&role={self.role}"
        
        # Add insecure mode
        if self.insecure_mode:
            connection_string += "&insecure_mode=true"
        
        engine = create_engine(connection_string)
        logger.info("✅ Created SQLAlchemy engine")
        return engine
    
    def test_connection(self):
        """Test connection"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_VERSION(), CURRENT_USER(), CURRENT_ACCOUNT()")
            version, user, account = cursor.fetchone()
            
            logger.info(f"✅ Connection test successful!")
            logger.info(f"   Version: {version}")
            logger.info(f"   User: {user}")
            logger.info(f"   Account: {account}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False