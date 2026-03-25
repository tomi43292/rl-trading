"""
Service layer for consuming Market Data API (https://api.marketdata.app/v1).
Implements caching with Redis to avoid excessive API calls.
"""
import logging
from decimal import Decimal
from datetime import datetime, timedelta

import httpx
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

from .models import Symbol, OHLCV, PriceSnapshot

logger = logging.getLogger(__name__)


class MarketDataAPIError(Exception):
    """Raised when the Market Data API returns an error."""
    pass


class RateLimitError(MarketDataAPIError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, reset_at: float = 0):
        self.reset_at = reset_at
        super().__init__(f"Rate limit exceeded. Resets at {reset_at}")


class MarketDataService:
    """
    Service that fetches stock data from the Market Data API.
    Uses Redis cache to minimize API calls and respect rate limits.
    """

    def __init__(self):
        self.base_url = settings.MARKETDATA_BASE_URL
        self.token = settings.MARKETDATA_API_TOKEN
        self.cache_ttl = settings.MARKETDATA_CACHE_TTL
        self._client = None

    @property
    def client(self) -> httpx.Client:
        """Persistent HTTP client for connection reuse."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={'Authorization': f'Bearer {self.token}'},
                timeout=httpx.Timeout(10.0, read=30.0),
            )
        return self._client

    def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def _request(self, path: str, params: dict = None) -> dict:
        """
        Make a cached request to the Market Data API.
        Returns parsed JSON response.
        """
        cache_key = f"marketdata:{path}:{params}"
        cached = cache.get(cache_key)
        if cached:
            logger.debug(f"Cache HIT: {path}")
            return cached

        logger.info(f"Cache MISS, fetching: {path} params={params}")
        try:
            response = self.client.get(path, params=params)
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
            raise MarketDataAPIError(f"Connection error: {e}") from e

        if response.status_code == 429:
            reset_at = response.headers.get('X-Api-Ratelimit-Reset', 0)
            raise RateLimitError(float(reset_at))

        if response.status_code >= 400:
            raise MarketDataAPIError(
                f"API error {response.status_code}: {response.text}"
            )

        data = response.json()
        if data.get('s') == 'error':
            raise MarketDataAPIError(f"API returned error: {data.get('errmsg', 'Unknown')}")

        cache.set(cache_key, data, self.cache_ttl)
        return data

    # ──────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────

    def get_price(self, symbol: str) -> dict:
        """
        Get the latest price for a symbol.
        Returns: {'symbol': 'AAPL', 'price': 150.50, 'change': 1.25, 'change_percent': 0.84, 'volume': 1000000}
        """
        data = self._request(f'/stocks/quotes/{symbol}/')
        if data.get('s') != 'ok':
            raise MarketDataAPIError(f"No data for {symbol}")

        return {
            'symbol': symbol.upper(),
            'price': data.get('mid', [None])[0] or data.get('last', [None])[0],
            'change': data.get('change', [None])[0],
            'change_percent': data.get('changepct', [None])[0],
            'volume': data.get('volume', [None])[0],
            'updated': data.get('updated', [None])[0],
        }

    def get_prices_bulk(self, symbols: list[str]) -> list[dict]:
        """Get latest prices for multiple symbols."""
        results = []
        for symbol in symbols:
            try:
                price = self.get_price(symbol)
                results.append(price)
            except MarketDataAPIError as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                results.append({'symbol': symbol, 'price': None, 'error': str(e)})
        return results

    def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        from_date: str = None,
        to_date: str = None,
        countback: int = None,
    ) -> list[dict]:
        """
        Get OHLCV candle data for a symbol.
        resolution: '1' (1min), '5', '15', '30', '60', 'D' (daily), 'W', 'M'
        """
        params = {'resolution': resolution}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if countback:
            params['countback'] = countback

        data = self._request(f'/stocks/candles/{resolution}/{symbol}/', params=params)
        if data.get('s') != 'ok':
            raise MarketDataAPIError(f"No candle data for {symbol}")

        candles = []
        timestamps = data.get('t', [])
        opens = data.get('o', [])
        highs = data.get('h', [])
        lows = data.get('l', [])
        closes = data.get('c', [])
        volumes = data.get('v', [])

        for i in range(len(timestamps)):
            candles.append({
                'timestamp': datetime.fromtimestamp(timestamps[i], tz=timezone.utc),
                'open': opens[i],
                'high': highs[i],
                'low': lows[i],
                'close': closes[i],
                'volume': volumes[i] if i < len(volumes) else 0,
            })

        return candles

    def get_market_status(self) -> dict:
        """Check if the US stock market is currently open."""
        data = self._request('/markets/status/')
        return data

    # ──────────────────────────────────────────────
    # Database persistence methods
    # ──────────────────────────────────────────────

    def sync_price_to_db(self, symbol_ticker: str) -> PriceSnapshot:
        """Fetch latest price and save to database."""
        price_data = self.get_price(symbol_ticker)
        symbol, _ = Symbol.objects.get_or_create(ticker=symbol_ticker.upper())

        snapshot, _ = PriceSnapshot.objects.update_or_create(
            symbol=symbol,
            defaults={
                'price': Decimal(str(price_data['price'])) if price_data['price'] else Decimal('0'),
                'change': Decimal(str(price_data['change'])) if price_data['change'] else None,
                'change_percent': Decimal(str(price_data['change_percent'])) if price_data['change_percent'] else None,
                'volume': price_data['volume'],
            }
        )
        logger.info(f"Synced price for {symbol_ticker}: ${snapshot.price}")
        return snapshot

    def sync_candles_to_db(
        self,
        symbol_ticker: str,
        resolution: str = 'D',
        countback: int = 365,
    ) -> int:
        """Fetch historical candles and save to database. Returns count of new records."""
        symbol, _ = Symbol.objects.get_or_create(ticker=symbol_ticker.upper())
        candles = self.get_candles(symbol_ticker, resolution=resolution, countback=countback)

        created_count = 0
        for candle in candles:
            _, created = OHLCV.objects.update_or_create(
                symbol=symbol,
                timestamp=candle['timestamp'],
                defaults={
                    'open': Decimal(str(candle['open'])),
                    'high': Decimal(str(candle['high'])),
                    'low': Decimal(str(candle['low'])),
                    'close': Decimal(str(candle['close'])),
                    'volume': candle['volume'],
                }
            )
            if created:
                created_count += 1

        logger.info(f"Synced {created_count} new candles for {symbol_ticker}")
        return created_count
