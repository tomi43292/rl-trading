"""Serializers for indicators app."""
from rest_framework import serializers


class IndicatorRequestSerializer(serializers.Serializer):
    """Validates input for indicator calculation endpoints."""
    symbol = serializers.CharField(max_length=10)
    limit = serializers.IntegerField(min_value=50, max_value=5000, default=500, required=False)

    def validate_symbol(self, value):
        return value.upper().strip()
