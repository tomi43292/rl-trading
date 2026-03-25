"""Admin configuration for trading models."""
from django.contrib import admin
from .models import TrainingSession, Trade


@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'status', 'episodes', 'initial_cash', 'profit_loss', 'profit_loss_pct', 'created_at']
    list_filter = ['status', 'symbol']
    readonly_fields = ['started_at', 'completed_at', 'created_at']


@admin.register(Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = ['session', 'symbol', 'trade_type', 'quantity', 'price', 'total_value', 'profit', 'step']
    list_filter = ['trade_type', 'symbol']
