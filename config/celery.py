"""Celery configuration for RL Trading project."""
import os
import warnings
from celery import Celery
from celery.schedules import crontab

# Suppress the loud OpenAI Gym deprecation warning in Celery worker subprocesses
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*Please upgrade to Gymnasium.*')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

app = Celery('rl_trading')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Periodic tasks
app.conf.beat_schedule = {
    'ingest-prices-every-minute': {
        'task': 'market_data.tasks.ingest_latest_prices',
        'schedule': crontab(minute='*', hour='9-16', day_of_week='1-5'),
        'args': [['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'SPY']],
    },
    'calculate-indicators-every-5min': {
        'task': 'indicators.tasks.calculate_all_indicators',
        'schedule': crontab(minute='*/5', hour='9-16', day_of_week='1-5'),
        'args': [['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'SPY']],
    },
}
