"""Models for market data ingestion and storage."""
from django.db import models


class Symbol(models.Model):
    """A tradeable stock symbol (e.g., AAPL, MSFT)."""
    ticker = models.CharField(max_length=10, unique=True, db_index=True)
    name = models.CharField(max_length=200, blank=True, default='')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['ticker']

    def __str__(self):
        return self.ticker


class OHLCV(models.Model):
    """Open-High-Low-Close-Volume candle data."""
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='candles')
    timestamp = models.DateTimeField(db_index=True)
    open = models.DecimalField(max_digits=12, decimal_places=4)
    high = models.DecimalField(max_digits=12, decimal_places=4)
    low = models.DecimalField(max_digits=12, decimal_places=4)
    close = models.DecimalField(max_digits=12, decimal_places=4)
    volume = models.BigIntegerField(default=0)

    class Meta:
        ordering = ['-timestamp']
        unique_together = ['symbol', 'timestamp']
        indexes = [
            models.Index(fields=['symbol', '-timestamp']),
        ]
        verbose_name = 'OHLCV Candle'
        verbose_name_plural = 'OHLCV Candles'

    def __str__(self):
        return f"{self.symbol.ticker} {self.timestamp:%Y-%m-%d %H:%M} C:{self.close}"


class PriceSnapshot(models.Model):
    """Latest price snapshot from Market Data API."""
    symbol = models.OneToOneField(Symbol, on_delete=models.CASCADE, related_name='latest_price')
    price = models.DecimalField(max_digits=12, decimal_places=4)
    change = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    change_percent = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Price Snapshot'

    def __str__(self):
        return f"{self.symbol.ticker}: ${self.price}"
