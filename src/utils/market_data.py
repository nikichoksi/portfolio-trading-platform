"""
Market data utilities.
Provides comprehensive list of tradeable stocks across all sectors.
"""

from typing import Dict, List

# Comprehensive list of popular stocks by sector
MARKET_STOCKS = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC",
        "ORCL", "ADBE", "CRM", "CSCO", "AVGO", "QCOM", "TXN", "IBM", "SNOW", "PLTR",
        "UBER", "LYFT", "SQ", "PYPL", "SHOP", "TWLO", "ZM", "DOCU", "CRWD", "NET",
        "DDOG", "MDB", "TEAM", "WDAY", "NOW", "PANW", "FTNT", "ZS", "OKTA", "MNDY"
    ],
    "Finance": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "COF",
        "USB", "PNC", "TFC", "BK", "STT", "ALLY", "SOFI", "V", "MA", "PYPL",
        "SQ", "COIN", "HOOD", "LC", "UPST", "AFRM", "NU", "MELI", "SE"
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "MRK", "DHR", "CVS", "BMY",
        "AMGN", "GILD", "CI", "ISRG", "REGN", "VRTX", "HUM", "ZTS", "BIIB", "ILMN",
        "MRNA", "DXCM", "ALGN", "IDXX", "IQV", "SYK", "BDX", "EW", "MTD", "A"
    ],
    "Consumer": [
        "AMZN", "TSLA", "NKE", "MCD", "SBUX", "HD", "LOW", "TGT", "WMT", "COST",
        "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "DISH", "PARA", "WBD",
        "NKE", "LULU", "ROST", "TJX", "DG", "DLTR", "BBY", "ETSY", "W", "CHWY"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
        "KMI", "WMB", "OKE", "LNG", "FANG", "DVN", "HES", "MRO", "APA", "CTRA"
    ],
    "Industrials": [
        "BA", "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE", "FDX",
        "UNP", "NSC", "CSX", "UBER", "LYFT", "DAL", "UAL", "AAL", "LUV", "JBLU"
    ],
    "Materials": [
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "VALE", "BHP", "RIO",
        "DD", "DOW", "PPG", "NUE", "STLD", "CLF", "X", "AA", "ALB", "SQM"
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "SPG", "O", "VICI",
        "AVB", "EQR", "MAA", "UDR", "CPT", "ESS", "AIV", "ELS", "SUI", "CUBE"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES",
        "ED", "ETR", "FE", "AES", "PPL", "CMS", "DTE", "EIX", "PEG", "AWK"
    ],
    "Communication Services": [
        "GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "DISH",
        "PARA", "WBD", "SPOT", "RBLX", "PINS", "SNAP", "TWTR", "MTCH", "BMBL", "YELP"
    ],
    "Crypto & Blockchain": [
        "COIN", "MARA", "RIOT", "HUT", "BITF", "MSTR", "SQ", "HOOD", "SI", "GBTC"
    ],
    "ETFs & Index": [
        "SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "VEA", "VWO", "AGG", "BND",
        "TLT", "GLD", "SLV", "USO", "UNG", "XLE", "XLF", "XLK", "XLV", "XLI"
    ]
}


def get_all_market_stocks() -> Dict[str, List[str]]:
    """Get all available market stocks grouped by sector"""
    return MARKET_STOCKS


def get_stock_sector(ticker: str) -> str:
    """Get sector for a given ticker"""
    for sector, tickers in MARKET_STOCKS.items():
        if ticker.upper() in tickers:
            return sector
    return "Other"


def get_popular_stocks(limit: int = 50) -> List[str]:
    """Get most popular stocks"""
    popular = []
    for sector in ["Technology", "Finance", "Healthcare", "Consumer"]:
        popular.extend(MARKET_STOCKS[sector][:12])
    return popular[:limit]


def search_stocks(query: str) -> List[tuple]:
    """Search for stocks by ticker or sector"""
    results = []
    query = query.upper()

    for sector, tickers in MARKET_STOCKS.items():
        for ticker in tickers:
            if query in ticker or query in sector.upper():
                results.append((ticker, sector))

    return results[:50]  # Limit results
