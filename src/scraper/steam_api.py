"""
Steam API client for scraping game reviews.
Handles communication with Steam's review endpoint.
"""
import requests
import logging
from typing import Optional, List
from datetime import datetime
from tqdm import tqdm

from src.scraper.data_models import ReviewBatch, SteamReview, ScraperStats
from src.scraper.utils import (
    rate_limiter,
    retry_with_backoff,
    save_checkpoint,
    load_checkpoint,
)
from src.utils.config import (
    STEAM_REVIEWS_ENDPOINT,
    USER_AGENT,
    REQUEST_TIMEOUT,
    DEFAULT_NUM_PER_PAGE,
    CHECKPOINT_INTERVAL,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamAPIError(Exception):
    """Custom exception for Steam API errors."""
    pass


class SteamReviewScraper:
    """
    Scraper for Steam game reviews.
    """

    def __init__(self, app_id: int):
        """
        Initialize the scraper.

        Args:
            app_id: Steam application ID
        """
        self.app_id = app_id
        self.endpoint = STEAM_REVIEWS_ENDPOINT.format(app_id=app_id)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.stats = ScraperStats()

    @rate_limiter(delay=0.5)
    @retry_with_backoff(exceptions=(requests.RequestException,))
    def fetch_review_page(
        self,
        cursor: str = "*",
        review_type: str = "all",
        language: str = "english",
        filter_type: str = "recent",
        num_per_page: int = DEFAULT_NUM_PER_PAGE,
        day_range: Optional[int] = None,
    ) -> ReviewBatch:
        """
        Fetch a single page of reviews from Steam API.

        Args:
            cursor: Pagination cursor ('*' for first page)
            review_type: 'all', 'positive', or 'negative'
            language: Language code (e.g., 'english', 'polish', 'all')
            filter_type: 'recent', 'updated', or 'all'
            num_per_page: Number of reviews per page (max 100)
            day_range: Limit to reviews from last N days (optional)

        Returns:
            ReviewBatch object with reviews and pagination info

        Raises:
            SteamAPIError: If the API request fails
        """
        params = {
            'json': 1,
            'cursor': cursor,
            'language': language,
            'review_type': review_type,
            'purchase_type': 'steam',
            'filter': filter_type,
            'num_per_page': min(num_per_page, 100),  # Steam max is 100
        }

        if day_range:
            params['day_range'] = day_range

        try:
            response = self.session.get(
                self.endpoint,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if data.get('success') != 1:
                raise SteamAPIError(f"API returned success={data.get('success')}")

            # Parse into Pydantic model
            batch = ReviewBatch(**data)

            if not batch.is_successful():
                raise SteamAPIError("API request was not successful")

            logger.debug(f"Fetched {len(batch)} reviews (cursor: {cursor[:20]}...)")
            return batch

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            self.stats.record_failure()
            raise
        except Exception as e:
            logger.error(f"Failed to parse API response: {e}")
            self.stats.record_failure()
            raise SteamAPIError(f"Failed to parse response: {e}")

    def scrape_reviews(
        self,
        max_reviews: int = 10000,
        review_type: str = "negative",
        language: str = "english",
        filter_type: str = "recent",
        save_checkpoints: bool = True,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        resume_from_checkpoint: bool = False,
    ) -> List[SteamReview]:
        """
        Scrape multiple pages of reviews.

        Args:
            max_reviews: Maximum number of reviews to scrape
            review_type: 'all', 'positive', or 'negative'
            language: Language code
            filter_type: 'recent', 'updated', or 'all'
            save_checkpoints: Whether to save periodic checkpoints
            checkpoint_interval: Save checkpoint every N reviews
            resume_from_checkpoint: Try to resume from existing checkpoint

        Returns:
            List of SteamReview objects
        """
        self.stats.start_time = datetime.now()
        reviews: List[SteamReview] = []
        cursor = "*"

        # Try to resume from checkpoint
        if resume_from_checkpoint:
            checkpoint_data = load_checkpoint("scraping", self.app_id)
            if checkpoint_data:
                reviews, cursor = checkpoint_data
                logger.info(f"Resuming from checkpoint with {len(reviews)} existing reviews")

        logger.info(f"Starting scrape for app_id={self.app_id}")
        logger.info(f"Target: {max_reviews} {review_type} reviews in {language}")

        # Progress bar
        pbar = tqdm(total=max_reviews, initial=len(reviews), desc="Scraping reviews")

        try:
            while len(reviews) < max_reviews:
                # Fetch next page
                batch = self.fetch_review_page(
                    cursor=cursor,
                    review_type=review_type,
                    language=language,
                    filter_type=filter_type,
                )

                # Add reviews from this batch
                new_reviews = batch.reviews
                if not new_reviews:
                    logger.info("No more reviews available")
                    break

                reviews.extend(new_reviews)
                self.stats.add_batch(batch)

                # Update progress bar
                pbar.update(len(new_reviews))

                # Save checkpoint if needed
                if save_checkpoints and len(reviews) % checkpoint_interval == 0:
                    save_checkpoint(
                        reviews=reviews,
                        checkpoint_name="scraping",
                        app_id=self.app_id,
                        cursor=batch.cursor,
                    )

                # Check if there are more pages
                if not batch.has_more_pages():
                    logger.info("Reached end of available reviews")
                    break

                cursor = batch.cursor

        except KeyboardInterrupt:
            logger.info("\nScraping interrupted by user")
            if save_checkpoints:
                save_checkpoint(
                    reviews=reviews,
                    checkpoint_name="scraping",
                    app_id=self.app_id,
                    cursor=cursor,
                )

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            if save_checkpoints:
                save_checkpoint(
                    reviews=reviews,
                    checkpoint_name="scraping",
                    app_id=self.app_id,
                    cursor=cursor,
                )
            raise

        finally:
            pbar.close()
            self.stats.end_time = datetime.now()

        # Trim to max_reviews
        reviews = reviews[:max_reviews]

        logger.info(f"\nScraping completed!")
        logger.info(f"Total reviews scraped: {len(reviews)}")

        return reviews

    def get_stats_summary(self) -> dict:
        """Get scraping statistics summary."""
        return self.stats.get_summary()


def quick_scrape(
    app_id: int,
    max_reviews: int = 1000,
    review_type: str = "negative",
    language: str = "english",
) -> List[SteamReview]:
    """
    Quick convenience function to scrape reviews.

    Args:
        app_id: Steam application ID
        max_reviews: Maximum number of reviews to scrape
        review_type: 'all', 'positive', or 'negative'
        language: Language code

    Returns:
        List of SteamReview objects
    """
    scraper = SteamReviewScraper(app_id)
    reviews = scraper.scrape_reviews(
        max_reviews=max_reviews,
        review_type=review_type,
        language=language,
        save_checkpoints=True,
    )

    # Print summary
    stats = scraper.get_stats_summary()
    logger.info("\n" + "=" * 50)
    logger.info("SCRAPING SUMMARY")
    logger.info("=" * 50)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    return reviews
