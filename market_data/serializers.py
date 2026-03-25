"""Serializers for market data API endpoints."""
from rest_framework import serializers
from .models import Symbol, OHLCV, PriceSnapshot


class SymbolSerializer(serializers.ModelSerializer):
    candle_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = Symbol
        fields = ['id', 'ticker', 'name', 'is_active', 'candle_count', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate_ticker(self, value):
        return value.upper().strip()


class OHLCVSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source='symbol.ticker', read_only=True)

    class Meta:
        model = OHLCV
        fields = ['id', 'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        read_only_fields = ['id']


class PriceSnapshotSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source='symbol.ticker', read_only=True)

    class Meta:
        model = PriceSnapshot
        fields = ['ticker', 'price', 'change', 'change_percent', 'volume', 'updated_at']


class IngestRequestSerializer(serializers.Serializer):
    """Validates input for ingestion endpoints."""
    symbols = serializers.ListField(
        child=serializers.CharField(max_length=10),
        min_length=1,
        max_length=20,
        help_text="List of ticker symbols to ingest"
    )
    resolution = serializers.ChoiceField(
        choices=['1', '5', '15', '30', '60', 'D', 'W', 'M'],
        default='D',
        required=False,
    )
    countback = serializers.IntegerField(
        min_value=1,
        max_value=5000,
        default=365,
        required=False,
    )

    def validate_symbols(self, value):
        return [s.upper().strip() for s in value]
