"""
Deep Q-Network (DQN) agent for stock trading.
Uses a neural network to approximate the Q-function and learn optimal
buy/sell/hold decisions based on technical indicators.
"""
import random
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Learning agent that learns to trade stocks.

    The agent uses an epsilon-greedy strategy to balance exploration
    (random actions) and exploitation (learned actions). The neural
    network is trained using experience replay from a memory buffer.
    """

    def __init__(self, state_size: int, action_space, memory_size: int = 3000):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_space.nvec.prod()

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

        # Hyperparameters
        self.gamma = 0.95          # Discount factor for future rewards
        self.epsilon = 1.0         # Exploration rate (starts at 100%)
        self.epsilon_min = 0.01    # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate per episode
        self.learning_rate = 0.001
        self.batch_size = 32

        # Build the neural network
        self.model = self._build_model()

    def _build_model(self):
        """Build the DQN neural network using TensorFlow/Keras."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                Dense(64, input_dim=self.state_size, activation='relu'),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(self.action_size, activation='linear'),
            ])
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            logger.info(f"DQN model built: state_size={self.state_size}, action_size={self.action_size}")
            return model
        except ImportError:
            logger.error("TensorFlow not available. Install with: pip install tensorflow")
            raise

    def remember(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action using epsilon-greedy strategy.
        With probability epsilon: random action (exploration).
        With probability 1-epsilon: best known action (exploitation).
        """
        if np.random.rand() <= self.epsilon:
            return (
                random.randrange(self.action_space.nvec[0]),
                random.randrange(self.action_space.nvec[1]),
            )

        act_values = self.model.predict(state, verbose=0)
        return np.unravel_index(np.argmax(act_values[0]), self.action_space.nvec)

    def replay(self, batch_size: int = None):
        """
        Train the model using a random batch from the experience replay buffer.
        This is the core learning step of DQN.
        """
        batch_size = batch_size or self.batch_size
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )

            target_f = self.model.predict(state, verbose=0)
            action_idx = np.ravel_multi_index(action, self.action_space.nvec)
            target_f[0][action_idx] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        """Save the trained model to disk."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load a trained model from disk."""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def get_config(self) -> dict:
        """Return agent configuration as a serializable dict."""
        return {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'gamma': self.gamma,
            'epsilon': round(self.epsilon, 4),
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'memory_size': self.memory.maxlen,
            'memory_used': len(self.memory),
        }
