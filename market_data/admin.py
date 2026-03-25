"""Admin configuration for market_data models."""
from django.contrib import admin
from .models import Symbol, OHLCV, PriceSnapshot


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = ['ticker', 'name', 'is_active', 'created_at']
    list_filter = ['is_active']
    search_fields = ['ticker', 'name']


@admin.register(OHLCV)
class OHLCVAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    list_filter = ['symbol']
    date_hierarchy = 'timestamp'


@admin.register(PriceSnapshot)
class PriceSnapshotAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'price', 'change', 'change_percent', 'volume', 'updated_at']
