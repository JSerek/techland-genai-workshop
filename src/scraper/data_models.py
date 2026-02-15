"""
Pydantic data models for Steam review data.
Provides validation and structured data handling.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union
from datetime import datetime


class SteamAuthor(BaseModel):
    """Model for review author information."""
    steamid: str
    num_games_owned: int
    num_reviews: int
    playtime_forever: int  # total playtime in minutes
    playtime_last_two_weeks: int  # playtime in last 2 weeks in minutes
    playtime_at_review: int  # playtime when review was written in minutes
    last_played: int  # unix timestamp


class SteamReview(BaseModel):
    """Model for a single Steam review."""
    model_config = ConfigDict(populate_by_name=True)

    recommendationid: str
    author: SteamAuthor
    language: str
    review: str  # the actual review text
    timestamp_created: int
    timestamp_updated: int
    voted_up: bool  # True = positive, False = negative
    votes_up: int
    votes_funny: int
    weighted_vote_score: Union[str, float, int]  # Steam's weighted score (can be string, float, or int)
    comment_count: int
    steam_purchase: bool
    received_for_free: bool
    written_during_early_access: bool

    # Computed fields (populated after validation)
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    playtime_hours: Optional[float] = None
    sentiment: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Compute derived fields after model initialization."""
        # Convert timestamps to datetime objects
        if self.timestamp_created:
            self.created_date = datetime.fromtimestamp(self.timestamp_created)
        if self.timestamp_updated:
            self.updated_date = datetime.fromtimestamp(self.timestamp_updated)

        # Convert playtime to hours
        if self.author and self.author.playtime_at_review:
            self.playtime_hours = round(self.author.playtime_at_review / 60, 1)

        # Set sentiment based on voted_up
        self.sentiment = "positive" if self.voted_up else "negative"

    def to_dict_simplified(self) -> dict:
        """Return a simplified dictionary for export."""
        return {
            "review_id": self.recommendationid,
            "author_id": self.author.steamid,
            "review_text": self.review,
            "language": self.language,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "sentiment": self.sentiment,
            "voted_up": self.voted_up,
            "votes_up": self.votes_up,
            "votes_funny": self.votes_funny,
            "playtime_hours": self.playtime_hours,
            "playtime_at_review_minutes": self.author.playtime_at_review,
            "steam_purchase": self.steam_purchase,
            "received_for_free": self.received_for_free,
            "early_access": self.written_during_early_access,
        }


class QuerySummary(BaseModel):
    """Model for Steam API query summary."""
    num_reviews: int
    review_score: Optional[int] = None
    review_score_desc: Optional[str] = None
    total_positive: Optional[int] = None
    total_negative: Optional[int] = None
    total_reviews: Optional[int] = None


class ReviewBatch(BaseModel):
    """Model for a batch of reviews returned by Steam API."""
    success: int  # 1 = success, 2 = error
    query_summary: QuerySummary
    reviews: list[SteamReview]
    cursor: Optional[str] = "*"  # cursor for pagination

    def __len__(self) -> int:
        """Return number of reviews in this batch."""
        return len(self.reviews)

    def is_successful(self) -> bool:
        """Check if the API request was successful."""
        return self.success == 1

    def has_more_pages(self) -> bool:
        """Check if there are more pages to fetch."""
        return self.cursor is not None and self.cursor != ""


class ScraperStats(BaseModel):
    """Model for tracking scraper statistics."""
    total_reviews_scraped: int = 0
    total_positive: int = 0
    total_negative: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_batch(self, batch: ReviewBatch) -> None:
        """Update stats with a new batch of reviews."""
        self.total_reviews_scraped += len(batch.reviews)
        self.total_positive += sum(1 for r in batch.reviews if r.voted_up)
        self.total_negative += sum(1 for r in batch.reviews if not r.voted_up)
        self.total_requests += 1

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failed_requests += 1

    def get_duration_seconds(self) -> Optional[float]:
        """Get duration of scraping in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_summary(self) -> dict:
        """Get summary statistics as dictionary."""
        duration = self.get_duration_seconds()
        return {
            "total_reviews": self.total_reviews_scraped,
            "positive_reviews": self.total_positive,
            "negative_reviews": self.total_negative,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round((self.total_requests - self.failed_requests) / max(self.total_requests, 1) * 100, 2),
            "duration_seconds": round(duration, 2) if duration else None,
            "reviews_per_second": round(self.total_reviews_scraped / duration, 2) if duration and duration > 0 else None,
        }
