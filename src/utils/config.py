"""
Configuration file for the Steam Review Scraper project.
Contains all constants and settings used across the project.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

# Steam API Configuration
DYING_LIGHT_BEAST_APP_ID = 3008130  # Dying Light 2: The Beast
STEAM_REVIEWS_ENDPOINT = "https://store.steampowered.com/appreviews/{app_id}"

# Scraping settings
TARGET_NEGATIVE_REVIEWS = 100_000
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # exponential backoff multiplier
REQUEST_TIMEOUT = 10  # seconds

# Checkpointing
CHECKPOINT_INTERVAL = 10_000  # save checkpoint every N reviews
CHECKPOINT_DIR = RAW_DATA_DIR / "checkpoints"

# Default scraping parameters
DEFAULT_LANGUAGE = "english"
DEFAULT_REVIEW_TYPE = "negative"  # 'all', 'positive', 'negative'
DEFAULT_FILTER = "recent"  # 'all', 'recent', 'updated'
DEFAULT_NUM_PER_PAGE = 100  # max 100 per Steam API

# User Agent for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Data export formats
SUPPORTED_EXPORT_FORMATS = ["json", "csv", "parquet"]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
