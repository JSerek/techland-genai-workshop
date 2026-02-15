"""
Utility functions for the Steam scraper.
Includes rate limiting, checkpointing, and data export functions.
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Callable, Any
from functools import wraps
import pandas as pd

from src.scraper.data_models import SteamReview
from src.utils.config import (
    RATE_LIMIT_DELAY,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    CHECKPOINT_DIR,
)

# Setup logging
logger = logging.getLogger(__name__)


def rate_limiter(delay: float = RATE_LIMIT_DELAY):
    """
    Decorator to add rate limiting between function calls.

    Args:
        delay: Seconds to wait between calls
    """
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]  # Use list to make it mutable in closure

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            if elapsed < delay:
                time.sleep(delay - elapsed)

            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry a function with exponential backoff on failure.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise

                    wait_time = backoff_factor ** retries
                    logger.warning(
                        f"Attempt {retries}/{max_retries} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)

            return func(*args, **kwargs)

        return wrapper
    return decorator


def save_checkpoint(
    reviews: List[SteamReview],
    checkpoint_name: str,
    app_id: int,
    cursor: Optional[str] = None,
) -> Path:
    """
    Save a checkpoint of scraped reviews.

    Args:
        reviews: List of SteamReview objects
        checkpoint_name: Name for the checkpoint file
        app_id: Steam app ID
        cursor: Current pagination cursor

    Returns:
        Path to the saved checkpoint file
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_file = CHECKPOINT_DIR / f"{checkpoint_name}_{app_id}.json"

    checkpoint_data = {
        "app_id": app_id,
        "total_reviews": len(reviews),
        "cursor": cursor,
        "timestamp": time.time(),
        "reviews": [r.model_dump() for r in reviews],
    }

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Checkpoint saved: {checkpoint_file} ({len(reviews)} reviews)")
    return checkpoint_file


def load_checkpoint(checkpoint_name: str, app_id: int) -> Optional[tuple[List[SteamReview], str]]:
    """
    Load a checkpoint of scraped reviews.

    Args:
        checkpoint_name: Name of the checkpoint file
        app_id: Steam app ID

    Returns:
        Tuple of (reviews list, cursor) or None if checkpoint doesn't exist
    """
    checkpoint_file = CHECKPOINT_DIR / f"{checkpoint_name}_{app_id}.json"

    if not checkpoint_file.exists():
        logger.info(f"No checkpoint found: {checkpoint_file}")
        return None

    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        reviews = [SteamReview(**r) for r in checkpoint_data['reviews']]
        cursor = checkpoint_data.get('cursor')

        logger.info(f"Checkpoint loaded: {checkpoint_file} ({len(reviews)} reviews)")
        return reviews, cursor

    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
        return None


def save_reviews_to_json(reviews: List[SteamReview], output_path: Path, simplified: bool = True) -> None:
    """
    Save reviews to JSON file.

    Args:
        reviews: List of SteamReview objects
        output_path: Path to output file
        simplified: If True, use simplified dict format
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if simplified:
        data = [r.to_dict_simplified() for r in reviews]
    else:
        data = [r.model_dump() for r in reviews]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(reviews)} reviews to {output_path}")


def save_reviews_to_csv(reviews: List[SteamReview], output_path: Path) -> None:
    """
    Save reviews to CSV file.

    Args:
        reviews: List of SteamReview objects
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict_simplified() for r in reviews]
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.info(f"Saved {len(reviews)} reviews to {output_path}")


def save_reviews_to_parquet(reviews: List[SteamReview], output_path: Path) -> None:
    """
    Save reviews to Parquet file (efficient binary format).

    Args:
        reviews: List of SteamReview objects
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict_simplified() for r in reviews]
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False, engine='pyarrow')

    logger.info(f"Saved {len(reviews)} reviews to {output_path}")


def save_to_formats(
    reviews: List[SteamReview],
    base_path: Path,
    formats: List[str] = ['json', 'csv', 'parquet']
) -> dict[str, Path]:
    """
    Save reviews to multiple formats.

    Args:
        reviews: List of SteamReview objects
        base_path: Base path without extension
        formats: List of formats to save ('json', 'csv', 'parquet')

    Returns:
        Dictionary mapping format to saved file path
    """
    saved_files = {}

    for fmt in formats:
        output_path = base_path.parent / f"{base_path.stem}.{fmt}"

        if fmt == 'json':
            save_reviews_to_json(reviews, output_path)
        elif fmt == 'csv':
            save_reviews_to_csv(reviews, output_path)
        elif fmt == 'parquet':
            save_reviews_to_parquet(reviews, output_path)
        else:
            logger.warning(f"Unsupported format: {fmt}")
            continue

        saved_files[fmt] = output_path

    return saved_files


def get_review_statistics(reviews: List[SteamReview]) -> dict:
    """
    Calculate basic statistics from reviews.

    Args:
        reviews: List of SteamReview objects

    Returns:
        Dictionary with statistics
    """
    if not reviews:
        return {}

    total = len(reviews)
    positive = sum(1 for r in reviews if r.voted_up)
    negative = total - positive

    playtimes = [r.playtime_hours for r in reviews if r.playtime_hours is not None]
    votes = [r.votes_up for r in reviews]

    stats = {
        "total_reviews": total,
        "positive_reviews": positive,
        "negative_reviews": negative,
        "positive_percentage": round(positive / total * 100, 2) if total > 0 else 0,
        "avg_playtime_hours": round(sum(playtimes) / len(playtimes), 2) if playtimes else 0,
        "median_playtime_hours": sorted(playtimes)[len(playtimes) // 2] if playtimes else 0,
        "avg_votes_up": round(sum(votes) / len(votes), 2) if votes else 0,
        "max_votes_up": max(votes) if votes else 0,
    }

    return stats
