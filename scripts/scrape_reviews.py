#!/usr/bin/env python3
"""
CLI script for scraping Steam reviews.

Usage:
    python scripts/scrape_reviews.py --app-id 3008130 --max-reviews 10000 --review-type negative

For help:
    python scripts/scrape_reviews.py --help
"""
import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.steam_api import SteamReviewScraper
from src.scraper.utils import save_to_formats, get_review_statistics
from src.utils.config import (
    DYING_LIGHT_BEAST_APP_ID,
    DEFAULT_LANGUAGE,
    DEFAULT_REVIEW_TYPE,
    RAW_DATA_DIR,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape Steam game reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape 10k negative reviews for Dying Light 2: The Beast
  python scripts/scrape_reviews.py --max-reviews 10000

  # Scrape positive reviews with custom app ID
  python scripts/scrape_reviews.py --app-id 123456 --review-type positive --max-reviews 5000

  # Resume from checkpoint
  python scripts/scrape_reviews.py --resume

  # Export only to JSON
  python scripts/scrape_reviews.py --formats json
        """
    )

    parser.add_argument(
        '--app-id',
        type=int,
        default=DYING_LIGHT_BEAST_APP_ID,
        help=f'Steam application ID (default: {DYING_LIGHT_BEAST_APP_ID} - Dying Light 2: The Beast)'
    )

    parser.add_argument(
        '--max-reviews',
        type=int,
        default=10000,
        help='Maximum number of reviews to scrape (default: 10000)'
    )

    parser.add_argument(
        '--review-type',
        type=str,
        choices=['all', 'positive', 'negative'],
        default=DEFAULT_REVIEW_TYPE,
        help=f'Type of reviews to scrape (default: {DEFAULT_REVIEW_TYPE})'
    )

    parser.add_argument(
        '--language',
        type=str,
        default=DEFAULT_LANGUAGE,
        help=f'Language of reviews (default: {DEFAULT_LANGUAGE})'
    )

    parser.add_argument(
        '--filter',
        type=str,
        choices=['all', 'recent', 'updated'],
        default='recent',
        help='Filter type (default: recent)'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        help='Custom output filename (without extension)'
    )

    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        choices=['json', 'csv', 'parquet'],
        default=['json', 'csv', 'parquet'],
        help='Output formats (default: all formats)'
    )

    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable checkpoint saving'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint'
    )

    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10000,
        help='Save checkpoint every N reviews (default: 10000)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Generate output filename
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"reviews_{args.app_id}_{args.review_type}_{args.language}"

    output_base = RAW_DATA_DIR / output_name

    logger.info("=" * 70)
    logger.info("STEAM REVIEW SCRAPER")
    logger.info("=" * 70)
    logger.info(f"App ID: {args.app_id}")
    logger.info(f"Max reviews: {args.max_reviews}")
    logger.info(f"Review type: {args.review_type}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Filter: {args.filter}")
    logger.info(f"Output formats: {', '.join(args.formats)}")
    logger.info(f"Checkpoints: {'disabled' if args.no_checkpoints else 'enabled'}")
    logger.info("=" * 70 + "\n")

    try:
        # Initialize scraper
        scraper = SteamReviewScraper(app_id=args.app_id)

        # Scrape reviews
        reviews = scraper.scrape_reviews(
            max_reviews=args.max_reviews,
            review_type=args.review_type,
            language=args.language,
            filter_type=args.filter,
            save_checkpoints=not args.no_checkpoints,
            checkpoint_interval=args.checkpoint_interval,
            resume_from_checkpoint=args.resume,
        )

        if not reviews:
            logger.warning("No reviews were scraped!")
            return

        # Get statistics
        stats = get_review_statistics(reviews)
        scraper_stats = scraper.get_stats_summary()

        # Save to requested formats
        logger.info(f"\nSaving reviews to {len(args.formats)} format(s)...")
        saved_files = save_to_formats(reviews, output_base, formats=args.formats)

        # Print summary
        print("\n" + "=" * 70)
        print("SCRAPING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nReview Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nScraper Statistics:")
        for key, value in scraper_stats.items():
            print(f"  {key}: {value}")

        print("\nSaved Files:")
        for fmt, path in saved_files.items():
            print(f"  {fmt.upper()}: {path}")

        print("=" * 70)

    except KeyboardInterrupt:
        logger.info("\n\nScraping interrupted by user. Progress has been saved.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n\nScraping failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
