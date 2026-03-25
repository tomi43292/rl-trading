"""
TensorBoard logging para el pipeline de entrenamiento RL.
Registra métricas por episodio, por paso de replay, y resultados de backtest.
"""
import os
import logging
from datetime import datetime

from django.conf import settings

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Registra métricas de entrenamiento RL en TensorBoard.

    Métricas registradas:
    - Por episodio: recompensa total, epsilon, valor del portfolio
    - Por paso de replay: loss promedio de la red neuronal
    - Post-backtest: P/L final, cantidad de operaciones, valor del portfolio
    - Histogramas: distribución de pesos de la red por episodio
    """

    def __init__(self, session_id: int, symbol: str):
        import tensorflow as tf
        self._tf = tf

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(
            settings.TENSORBOARD_LOG_DIR,
            f'session_{session_id}_{symbol}_{timestamp}',
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self._writer = tf.summary.create_file_writer(self.log_dir)
        self._replay_step = 0

        logger.info(f"TensorBoard logger inicializado: {self.log_dir}")

    def log_episode(
        self,
        episode: int,
        reward: float,
        epsilon: float,
        portfolio_value: float,
    ) -> None:
        """Registra métricas al finalizar un episodio de entrenamiento."""
        with self._writer.as_default():
            self._tf.summary.scalar('entrenamiento/recompensa', reward, step=episode)
            self._tf.summary.scalar('entrenamiento/epsilon', epsilon, step=episode)
            self._tf.summary.scalar('entrenamiento/valor_portfolio', portfolio_value, step=episode)

    def log_replay(self, loss: float) -> None:
        """Registra la loss del replay de experiencia."""
        with self._writer.as_default():
            self._tf.summary.scalar('red_neuronal/loss', loss, step=self._replay_step)
        self._replay_step += 1

    def log_model_weights(self, model, episode: int) -> None:
        """Registra histogramas de los pesos de la red neuronal."""
        with self._writer.as_default():
            for layer in model.layers:
                for weight in layer.weights:
                    self._tf.summary.histogram(
                        f'pesos/{weight.name}', weight, step=episode,
                    )

    def log_backtest(
        self,
        portfolio_value: float,
        profit_loss: float,
        total_trades: int,
        buy_trades: int,
        sell_trades: int,
    ) -> None:
        """Registra resultados del backtesting."""
        with self._writer.as_default():
            self._tf.summary.scalar('backtest/valor_portfolio', portfolio_value, step=0)
            self._tf.summary.scalar('backtest/ganancia_perdida', profit_loss, step=0)
            self._tf.summary.scalar('backtest/total_operaciones', total_trades, step=0)
            self._tf.summary.scalar('backtest/compras', buy_trades, step=0)
            self._tf.summary.scalar('backtest/ventas', sell_trades, step=0)

    def flush(self) -> None:
        """Fuerza la escritura de las métricas pendientes a disco."""
        self._writer.flush()

    def close(self) -> None:
        """Cierra el writer y libera recursos."""
        self._writer.flush()
        self._writer.close()
        logger.info(f"TensorBoard logger cerrado: {self.log_dir}")
