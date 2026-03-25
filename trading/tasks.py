"""Celery tasks for async RL training."""
import logging
from celery import shared_task
from .services import TradingService

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    time_limit=600,
    soft_time_limit=540,
)
def train_agent_async(self, symbol: str, episodes: int = 100, batch_size: int = 32, initial_cash: float = 10000):
    """
    Train a DQN agent asynchronously via Celery.
    This allows training to happen in the background without blocking the API.
    """
    logger.info(f"Celery task started: training {symbol} for {episodes} episodes")
    try:
        session = TradingService.train_agent(
            symbol_ticker=symbol,
            episodes=episodes,
            batch_size=batch_size,
            initial_cash=initial_cash,
        )
        logger.info(f"Training complete: session {session.id}, P/L: {session.profit_loss}")
        return {
            'session_id': session.id,
            'status': session.status,
            'profit_loss': str(session.profit_loss),
            'profit_loss_pct': session.profit_loss_pct,
        }
    except Exception as e:
        logger.exception(f"Training task failed for {symbol}: {e}")
        raise
