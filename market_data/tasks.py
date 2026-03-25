"""Celery tasks for periodic market data ingestion."""
import logging
from celery import shared_task
from .services import MarketDataService, MarketDataAPIError, RateLimitError

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def ingest_latest_prices(self, symbols: list[str]):
    """
    Fetch and store latest prices for given symbols.
    Scheduled to run every minute during market hours.
    """
    service = MarketDataService()
    results = {'success': 0, 'errors': 0}

    try:
        for ticker in symbols:
            try:
                service.sync_price_to_db(ticker)
                results['success'] += 1
            except RateLimitError as e:
                logger.warning(f"Rate limited, will retry: {e}")
                raise self.retry(exc=e, countdown=60)
            except MarketDataAPIError as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                results['errors'] += 1
    finally:
        service.close()

    logger.info(f"Price ingestion complete: {results}")
    return results


@shared_task(
    bind=True,
    max_retries=2,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
)
def ingest_historical_candles(self, symbol: str, resolution: str = 'D', countback: int = 365):
    """
    Fetch and store historical OHLCV candles for a symbol.
    Triggered manually or on first setup.
    """
    service = MarketDataService()
    try:
        count = service.sync_candles_to_db(symbol, resolution, countback)
        logger.info(f"Ingested {count} candles for {symbol}")
        return {'symbol': symbol, 'new_candles': count}
    except MarketDataAPIError as e:
        logger.error(f"Historical ingestion failed for {symbol}: {e}")
        raise
    finally:
        service.close()
