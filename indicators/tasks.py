"""Celery tasks for indicator calculations."""
import logging
from celery import shared_task
from .services import IndicatorService

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2)
def calculate_all_indicators(self, symbols: list[str]):
    """
    Calculate and cache indicators for given symbols.
    Scheduled every 5 minutes during market hours.
    """
    results = {'success': 0, 'errors': 0}

    for ticker in symbols:
        try:
            df = IndicatorService.calculate_all(ticker)
            results['success'] += 1
            logger.info(f"Indicators calculated for {ticker}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Indicator calculation failed for {ticker}: {e}")
            results['errors'] += 1

    logger.info(f"Indicator batch complete: {results}")
    return results
