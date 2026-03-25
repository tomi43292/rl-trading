"""Tests for the trading app."""
from decimal import Decimal
from datetime import timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APITestCase
from rest_framework import status

from market_data.models import Symbol, OHLCV
from .models import TrainingSession, Trade
from .environment import StockTradingEnv


class StockTradingEnvTest(TestCase):
    """Tests for the RL gym environment."""

    def setUp(self):
        """Create a simple DataFrame with indicator columns for testing."""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'close': np.cumsum(np.random.randn(n)) + 150,
            'EMA7': np.cumsum(np.random.randn(n)) + 150,
            'EMA14': np.cumsum(np.random.randn(n)) + 150,
            'EMA50': np.cumsum(np.random.randn(n)) + 150,
            'EMA200': np.cumsum(np.random.randn(n)) + 150,
            'MACD_line': np.random.randn(n) * 0.5,
            'MACD_signal': np.random.randn(n) * 0.3,
            'MACD_diff': np.random.randn(n) * 0.2,
            'RSI': np.random.uniform(20, 80, n),
            'OBV': np.cumsum(np.random.randint(-1000, 1000, n)),
            'BB_high': np.ones(n),
            'BB_low': np.zeros(n),
        })

    def test_env_initialization(self):
        env = StockTradingEnv(self.df)
        self.assertEqual(env.initial_cash, 10000.0)
        self.assertEqual(env.shares_held, 0)
        self.assertEqual(env.current_step, 0)

    def test_env_reset(self):
        env = StockTradingEnv(self.df)
        obs = env.reset()
        self.assertEqual(len(obs), 12)  # 12 indicator columns
        self.assertEqual(env.cash, 10000.0)
        self.assertEqual(env.shares_held, 0)

    def test_buy_action(self):
        env = StockTradingEnv(self.df)
        env.reset()
        action = (0, 5)  # BUY 5 shares
        obs, reward, done, info = env.step(action)
        self.assertEqual(env.shares_held, 5)
        self.assertLess(env.cash, 10000.0)

    def test_sell_action(self):
        env = StockTradingEnv(self.df)
        env.reset()
        env.step((0, 5))  # BUY 5
        env.step((1, 3))  # SELL 3
        self.assertEqual(env.shares_held, 2)

    def test_sell_more_than_held(self):
        env = StockTradingEnv(self.df)
        env.reset()
        env.step((0, 3))  # BUY 3
        env.step((1, 9))  # SELL 9, but only 3 held
        self.assertEqual(env.shares_held, 0)  # sold max 3

    def test_hold_action(self):
        env = StockTradingEnv(self.df)
        env.reset()
        initial_cash = env.cash
        env.step((2, 0))  # HOLD
        self.assertEqual(env.cash, initial_cash)
        self.assertEqual(env.shares_held, 0)

    def test_episode_completes(self):
        env = StockTradingEnv(self.df)
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
        self.assertEqual(steps, len(self.df) - 1)

    def test_portfolio_value(self):
        env = StockTradingEnv(self.df, initial_cash=10000)
        env.reset()
        self.assertAlmostEqual(env.portfolio_value, 10000.0, places=0)

    def test_trade_history_recorded(self):
        env = StockTradingEnv(self.df)
        env.reset()
        env.step((0, 5))  # BUY
        env.step((1, 5))  # SELL
        self.assertEqual(len(env.trade_history), 2)
        self.assertEqual(env.trade_history[0]['type'], 'BUY')
        self.assertEqual(env.trade_history[1]['type'], 'SELL')


class TrainingSessionModelTest(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(ticker='AAPL', name='Apple Inc.')

    def test_create_session(self):
        session = TrainingSession.objects.create(
            symbol=self.symbol,
            episodes=50,
            batch_size=32,
        )
        self.assertEqual(session.status, 'PENDING')
        self.assertIn('10000', str(session.initial_cash))

    def test_session_str(self):
        session = TrainingSession.objects.create(symbol=self.symbol, episodes=100)
        self.assertIn('AAPL', str(session))
        self.assertIn('PENDING', str(session))


class TradeModelTest(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(ticker='AAPL')
        self.session = TrainingSession.objects.create(symbol=self.symbol, episodes=10)

    def test_create_trade(self):
        trade = Trade.objects.create(
            session=self.session,
            symbol=self.symbol,
            trade_type='BUY',
            quantity=10,
            price=Decimal('150.50'),
            total_value=Decimal('1505.00'),
            step=1,
        )
        self.assertIn('BUY', str(trade))
        self.assertIn('AAPL', str(trade))

    def test_auto_total_value(self):
        trade = Trade(
            session=self.session,
            symbol=self.symbol,
            trade_type='BUY',
            quantity=10,
            price=Decimal('150.50'),
            step=1,
        )
        trade.save()
        self.assertEqual(trade.total_value, Decimal('1505.00'))


class TrainingSessionAPITest(APITestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(ticker='AAPL')
        self.session = TrainingSession.objects.create(
            symbol=self.symbol,
            status='COMPLETED',
            episodes=50,
            final_portfolio_value=Decimal('11500.00'),
            profit_loss=Decimal('1500.00'),
            profit_loss_pct=15.0,
        )

    def test_list_sessions(self):
        response = self.client.get('/api/trading/sessions/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_retrieve_session(self):
        response = self.client.get(f'/api/trading/sessions/{self.session.id}/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['ticker'], 'AAPL')
        self.assertEqual(response.data['status'], 'COMPLETED')

    def test_filter_by_symbol(self):
        response = self.client.get('/api/trading/sessions/?symbol=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_filter_by_status(self):
        response = self.client.get('/api/trading/sessions/?status=COMPLETED')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_session_summary(self):
        Trade.objects.create(
            session=self.session, symbol=self.symbol,
            trade_type='BUY', quantity=10, price=Decimal('150'), total_value=Decimal('1500'), step=1
        )
        response = self.client.get(f'/api/trading/sessions/{self.session.id}/summary/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['total_trades'], 1)


class TrainViewTest(APITestCase):
    def test_train_missing_symbol(self):
        response = self.client.post('/api/trading/train/', {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_train_invalid_episodes(self):
        response = self.client.post('/api/trading/train/', {
            'symbol': 'AAPL', 'episodes': 5,  # below minimum 10
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('trading.views.train_agent_async')
    def test_train_dispatches_celery_task(self, mock_task):
        mock_task.delay.return_value = MagicMock(id='test-task-id')
        response = self.client.post('/api/trading/train/', {
            'symbol': 'AAPL',
            'episodes': 50,
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertEqual(response.data['symbol'], 'AAPL')
        mock_task.delay.assert_called_once()
