"""Serializers for the trading app."""
from rest_framework import serializers
from .models import TrainingSession, Trade


class TradeSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source='symbol.ticker', read_only=True)

    class Meta:
        model = Trade
        fields = ['id', 'ticker', 'trade_type', 'quantity', 'price', 'total_value', 'profit', 'step', 'timestamp']


class TrainingSessionSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source='symbol.ticker', read_only=True)
    trades = TradeSerializer(many=True, read_only=True)

    class Meta:
        model = TrainingSession
        fields = [
            'id', 'ticker', 'status', 'episodes', 'batch_size', 'initial_cash',
            'final_epsilon', 'total_reward', 'final_portfolio_value',
            'profit_loss', 'profit_loss_pct', 'model_path',
            'started_at', 'completed_at', 'created_at', 'error_message',
            'trades',
        ]


class TrainingSessionListSerializer(serializers.ModelSerializer):
    """Lighter serializer for list view (no trades)."""
    ticker = serializers.CharField(source='symbol.ticker', read_only=True)
    trades_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = TrainingSession
        fields = [
            'id', 'ticker', 'status', 'episodes', 'initial_cash',
            'final_portfolio_value', 'profit_loss', 'profit_loss_pct',
            'trades_count', 'started_at', 'completed_at', 'created_at',
        ]


class TrainRequestSerializer(serializers.Serializer):
    """Validates input for training endpoint."""
    symbol = serializers.CharField(max_length=10)
    episodes = serializers.IntegerField(min_value=10, max_value=1000, default=100)
    batch_size = serializers.IntegerField(min_value=8, max_value=128, default=32)
    initial_cash = serializers.FloatField(min_value=100, max_value=1000000, default=10000)

    def validate_symbol(self, value):
        return value.upper().strip()
