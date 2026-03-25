"""Models for the trading app: trades, positions, and training sessions."""
from django.db import models
from market_data.models import Symbol


class TrainingSession(models.Model):
    """Tracks RL agent training runs."""
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='training_sessions')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    episodes = models.IntegerField(default=100)
    batch_size = models.IntegerField(default=32)
    initial_cash = models.DecimalField(max_digits=12, decimal_places=2, default=10000)

    # Results
    final_epsilon = models.FloatField(null=True, blank=True)
    total_reward = models.FloatField(null=True, blank=True)
    final_portfolio_value = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    profit_loss = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    profit_loss_pct = models.FloatField(null=True, blank=True)
    model_path = models.CharField(max_length=500, blank=True, default='')

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Training {self.symbol.ticker} [{self.status}] - {self.episodes} episodes"


class Trade(models.Model):
    """Individual trades executed by the RL agent during backtesting."""
    TYPE_CHOICES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    ]

    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='trades')
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='trades')
    trade_type = models.CharField(max_length=4, choices=TYPE_CHOICES)
    quantity = models.IntegerField()
    price = models.DecimalField(max_digits=12, decimal_places=4)
    total_value = models.DecimalField(max_digits=12, decimal_places=2)
    profit = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    step = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['step']

    def __str__(self):
        return f"{self.trade_type} {self.quantity}x {self.symbol.ticker} @ ${self.price}"

    def save(self, *args, **kwargs):
        if not self.total_value:
            self.total_value = self.quantity * self.price
        super().save(*args, **kwargs)
