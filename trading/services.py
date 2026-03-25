"""
Trading service: orchestrates RL training and backtesting.
Connects market_data, indicators, and the RL agent.
"""
import logging
import os
from decimal import Decimal
from datetime import datetime

import numpy as np
from django.conf import settings
from django.utils import timezone

from indicators.services import IndicatorService
from market_data.models import Symbol
from .models import TrainingSession, Trade
from .environment import StockTradingEnv
from .agent import DQNAgent
from .callbacks import TrainingLogger

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(settings.BASE_DIR, 'trained_models')
os.makedirs(MODELS_DIR, exist_ok=True)


class TradingService:
    """
    Orchestrates RL agent training and backtesting using real market data.
    """

    @staticmethod
    def train_agent(
        symbol_ticker: str,
        episodes: int = 100,
        batch_size: int = 32,
        initial_cash: float = 10000.0,
        test_split: float = 0.1,
    ) -> TrainingSession:
        """
        Train a DQN agent on historical data for a given symbol.

        Steps:
        1. Load OHLCV data and calculate indicators via IndicatorService
        2. Split into train/test sets
        3. Create gym environment and DQN agent
        4. Train for N episodes
        5. Backtest on test set
        6. Save model and results to DB
        """
        symbol, _ = Symbol.objects.get_or_create(ticker=symbol_ticker.upper())

        session = TrainingSession.objects.create(
            symbol=symbol,
            status='RUNNING',
            episodes=episodes,
            batch_size=batch_size,
            initial_cash=Decimal(str(initial_cash)),
            started_at=timezone.now(),
        )

        tb_logger = None
        try:
            # Step 1: Get data with indicators
            logger.info(f"Loading indicator data for {symbol_ticker}...")
            df = IndicatorService.calculate_all(symbol_ticker, limit=2000)

            if len(df) < 100:
                raise ValueError(f"Not enough data for {symbol_ticker}: {len(df)} rows (need 100+)")

            # Step 2: Train/test split
            split_idx = int(len(df) * (1 - test_split))
            train_data = df.iloc[:split_idx].copy()
            test_data = df.iloc[split_idx:].copy()

            logger.info(f"Data split: train={len(train_data)}, test={len(test_data)}")

            # Step 3: Create environment, agent, and TensorBoard logger
            env = StockTradingEnv(train_data, initial_cash=initial_cash)
            state_size = env.observation_space.shape[0]
            agent = DQNAgent(state_size, env.action_space)
            tb_logger = TrainingLogger(session.id, symbol_ticker)

            # Step 4: Training loop
            logger.info(f"Starting training: {episodes} episodes...")
            episode_rewards = []

            for e in range(episodes):
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                total_reward = 0

                while True:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    next_state = np.reshape(next_state, [1, state_size])
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    if done:
                        break

                episode_rewards.append(total_reward)

                # Replay y registrar loss en TensorBoard
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    if loss is not None:
                        tb_logger.log_replay(loss)

                # Registrar métricas del episodio
                tb_logger.log_episode(e, total_reward, agent.epsilon, env.portfolio_value)

                # Registrar histogramas de pesos cada 10 episodios
                if (e + 1) % 10 == 0:
                    tb_logger.log_model_weights(agent.model, e)
                    logger.info(
                        f"Episode {e+1}/{episodes} | "
                        f"Reward: {total_reward:.2f} | "
                        f"Epsilon: {agent.epsilon:.3f}"
                    )

            # Step 5: Backtest on test data
            logger.info("Running backtest on test data...")
            backtest_result = TradingService._backtest(
                agent, test_data, state_size, initial_cash, session, symbol
            )

            # Registrar resultados del backtest en TensorBoard
            tb_logger.log_backtest(
                portfolio_value=backtest_result['portfolio_value'],
                profit_loss=backtest_result['portfolio_value'] - initial_cash,
                total_trades=backtest_result['trades_count'],
                buy_trades=len([t for t in backtest_result.get('trade_history', []) if t.get('type') == 'BUY']),
                sell_trades=len([t for t in backtest_result.get('trade_history', []) if t.get('type') == 'SELL']),
            )

            # Step 6: Save model
            model_filename = f"dqn_{symbol_ticker}_{session.id}.keras"
            model_path = os.path.join(MODELS_DIR, model_filename)
            agent.save(model_path)

            # Update session with results
            session.status = 'COMPLETED'
            session.completed_at = timezone.now()
            session.final_epsilon = agent.epsilon
            session.total_reward = sum(episode_rewards)
            session.final_portfolio_value = Decimal(str(round(backtest_result['portfolio_value'], 2)))
            session.profit_loss = session.final_portfolio_value - session.initial_cash
            session.profit_loss_pct = float(
                (session.final_portfolio_value - session.initial_cash) / session.initial_cash * 100
            )
            session.model_path = model_path
            session.tensorboard_log_dir = tb_logger.log_dir
            session.save()

            logger.info(
                f"Training complete for {symbol_ticker}: "
                f"P/L: ${session.profit_loss} ({session.profit_loss_pct:.2f}%)"
            )
            return session

        except Exception as e:
            session.status = 'FAILED'
            session.error_message = str(e)
            session.completed_at = timezone.now()
            session.save()
            logger.exception(f"Training failed for {symbol_ticker}: {e}")
            raise

        finally:
            if tb_logger is not None:
                tb_logger.close()

    @staticmethod
    def _backtest(
        agent: DQNAgent,
        test_data,
        state_size: int,
        initial_cash: float,
        session: TrainingSession,
        symbol: Symbol,
    ) -> dict:
        """Run the trained agent on test data and record trades."""
        test_env = StockTradingEnv(test_data, initial_cash=initial_cash)
        state = test_env.reset()
        state = np.reshape(state, [1, state_size])

        # Disable exploration during backtest
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = test_env.step(action)
            state = np.reshape(next_state, [1, state_size])

            if done:
                break

        # Save trades to DB
        for trade_record in test_env.trade_history:
            Trade.objects.create(
                session=session,
                symbol=symbol,
                trade_type=trade_record['type'],
                quantity=trade_record['shares'],
                price=Decimal(str(round(trade_record['price'], 4))),
                total_value=Decimal(str(round(trade_record['price'] * trade_record['shares'], 2))),
                profit=Decimal(str(round(trade_record.get('profit', 0), 2))),
                step=trade_record['step'],
            )

        agent.epsilon = original_epsilon

        return {
            'portfolio_value': test_env.portfolio_value,
            'cash': test_env.cash,
            'shares_held': test_env.shares_held,
            'total_profit': test_env.total_profit,
            'trades_count': len(test_env.trade_history),
        }

    @staticmethod
    def get_session_summary(session_id: int) -> dict:
        """Get detailed summary of a training session."""
        session = TrainingSession.objects.select_related('symbol').get(id=session_id)
        trades = Trade.objects.filter(session=session).order_by('step')

        buy_trades = trades.filter(trade_type='BUY')
        sell_trades = trades.filter(trade_type='SELL')

        return {
            'session_id': session.id,
            'symbol': session.symbol.ticker,
            'status': session.status,
            'episodes': session.episodes,
            'initial_cash': str(session.initial_cash),
            'final_portfolio_value': str(session.final_portfolio_value),
            'profit_loss': str(session.profit_loss),
            'profit_loss_pct': session.profit_loss_pct,
            'total_trades': trades.count(),
            'buy_trades': buy_trades.count(),
            'sell_trades': sell_trades.count(),
            'started_at': session.started_at.isoformat() if session.started_at else None,
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
            'duration_seconds': (
                (session.completed_at - session.started_at).total_seconds()
                if session.completed_at and session.started_at else None
            ),
        }
