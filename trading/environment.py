"""
Custom OpenAI Gym environment for stock trading with reinforcement learning.
Uses technical indicators from the indicators app as observation space.
"""
import warnings

# gym 0.26.2 is unmaintained but compatible with numpy 1.26.2.
# Suppress the deprecation spam before import so it doesn't flood logs.
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*Please upgrade to Gymnasium.*')

import gym
import numpy as np
import pandas as pd


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning.

    Observation space: OHLCV data + technical indicators (EMA, RSI, MACD, BB, OBV).
    Action space: [action_type (buy/sell/hold), amount (0-9 shares)].
    Reward: change in portfolio value per step.

    The environment simulates buying and selling a single stock using real
    market data fetched from the Market Data API via the indicators service.
    """

    metadata = {'render.modes': ['human']}

    # Indicator columns expected in the DataFrame
    INDICATOR_COLS = [
        'close', 'EMA7', 'EMA14', 'EMA50', 'EMA200',
        'MACD_line', 'MACD_signal', 'MACD_diff',
        'RSI', 'OBV', 'BB_high', 'BB_low',
    ]

    def __init__(self, data: pd.DataFrame, initial_cash: float = 10000.0, commission: float = 0.0):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.commission = commission
        self.end_step = len(self.data) - 1

        # Action: [type (0=buy, 1=sell, 2=hold), shares (0-9)]
        self.action_space = gym.spaces.MultiDiscrete([3, 10])

        # Observation: vector of indicator values at current step
        obs_size = len(self.INDICATOR_COLS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.buy_price = 0.0
        self.total_profit = 0.0
        self.done = False
        self.trade_history = []
        return self._get_obs()

    def step(self, action):
        """Execute one step in the environment."""
        action_type, amount = action
        current_price = float(self.data['close'].iloc[self.current_step])

        # Execute action
        if action_type == 0:  # BUY
            cost = current_price * amount * (1 + self.commission)
            if self.cash >= cost and amount > 0:
                self.cash -= cost
                self.shares_held += amount
                self.buy_price = current_price
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'BUY',
                    'shares': amount,
                    'price': current_price,
                })

        elif action_type == 1:  # SELL
            shares_to_sell = min(self.shares_held, amount)
            if shares_to_sell > 0:
                revenue = current_price * shares_to_sell * (1 - self.commission)
                self.cash += revenue
                profit = (current_price - self.buy_price) * shares_to_sell
                self.total_profit += profit
                self.shares_held -= shares_to_sell
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'profit': profit,
                })

        # else: action_type == 2 → HOLD (do nothing)

        # Advance step
        self.current_step += 1
        done = self.current_step >= self.end_step

        # Calculate reward as change in portfolio value
        if not done:
            next_price = float(self.data['close'].iloc[self.current_step])
        else:
            next_price = current_price

        portfolio_now = self.cash + (self.shares_held * next_price)
        portfolio_prev = self.cash + (self.shares_held * current_price)
        reward = portfolio_now - portfolio_prev

        return self._get_obs(), reward, done, {
            'portfolio_value': portfolio_now,
            'cash': self.cash,
            'shares_held': self.shares_held,
            'total_profit': self.total_profit,
        }

    def _get_obs(self):
        """Build the observation vector from current step's indicator values."""
        row = self.data.iloc[self.current_step]
        obs = np.array([
            float(row.get(col, 0.0)) for col in self.INDICATOR_COLS
        ], dtype=np.float32)
        return obs

    @property
    def portfolio_value(self):
        """Current total portfolio value."""
        current_price = float(self.data['close'].iloc[self.current_step])
        return self.cash + (self.shares_held * current_price)

    def render(self, mode='human'):
        """Print current state."""
        price = float(self.data['close'].iloc[self.current_step])
        print(
            f"Step {self.current_step}/{self.end_step} | "
            f"Price: ${price:.2f} | Cash: ${self.cash:.2f} | "
            f"Shares: {self.shares_held} | "
            f"Portfolio: ${self.portfolio_value:.2f} | "
            f"Profit: ${self.total_profit:.2f}"
        )
