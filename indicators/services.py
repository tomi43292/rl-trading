"""
Technical indicators calculation service using Pandas.
Calculates EMA, RSI, MACD, Bollinger Bands, OBV from OHLCV data.
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np
import ta

from market_data.models import Symbol, OHLCV

logger = logging.getLogger(__name__)


class IndicatorService:
    """
    Calculates technical indicators from stored OHLCV data using Pandas.
    Uses the 'ta' library for standardized indicator calculations.
    """

    @staticmethod
    def get_dataframe(symbol_ticker: str, limit: int = 500) -> pd.DataFrame:
        """
        Load OHLCV data from DB into a Pandas DataFrame.
        Returns DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        try:
            symbol = Symbol.objects.get(ticker=symbol_ticker.upper())
        except Symbol.DoesNotExist:
            raise ValueError(f"Symbol not found: {symbol_ticker}")

        candles = OHLCV.objects.filter(
            symbol=symbol
        ).order_by('timestamp').values(
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        )[:limit]

        if not candles:
            raise ValueError(f"No candle data for {symbol_ticker}")

        df = pd.DataFrame(list(candles))
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Loaded {len(df)} candles for {symbol_ticker}")
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        if windows is None:
            windows = [7, 14, 50, 200]

        for w in windows:
            col_name = f'EMA{w}'
            df[col_name] = ta.trend.EMAIndicator(
                close=df['close'], window=w, fillna=False
            ).ema_indicator()

        return df

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        macd = ta.trend.MACD(
            close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False
        )
        df['MACD_line'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        df['RSI'] = ta.momentum.RSIIndicator(
            close=df['close'], window=window, fillna=False
        ).rsi()
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands (high/low indicators)."""
        bb = ta.volatility.BollingerBands(
            close=df['close'], window=window, window_dev=std_dev, fillna=False
        )
        df['BB_high'] = bb.bollinger_hband_indicator()
        df['BB_low'] = bb.bollinger_lband_indicator()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add OBV (On-Balance Volume)."""
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'], volume=df['volume'], fillna=False
        ).on_balance_volume()
        return df

    @classmethod
    def calculate_all(cls, symbol_ticker: str, limit: int = 500) -> pd.DataFrame:
        """
        Calculate all technical indicators for a symbol.
        Returns a DataFrame with OHLCV + all indicators, NaN rows dropped.
        """
        df = cls.get_dataframe(symbol_ticker, limit)
        df = cls.add_ema(df)
        df = cls.add_macd(df)
        df = cls.add_rsi(df)
        df = cls.add_bollinger_bands(df)
        df = cls.add_obv(df)
        df = df.dropna()

        logger.info(
            f"Calculated indicators for {symbol_ticker}: "
            f"{len(df)} rows, {len(df.columns)} columns"
        )
        return df

    @classmethod
    def get_indicator_summary(cls, symbol_ticker: str) -> dict:
        """
        Get a summary of the latest indicator values for a symbol.
        Useful for quick dashboard display.
        """
        df = cls.calculate_all(symbol_ticker, limit=250)
        if df.empty:
            return {'symbol': symbol_ticker, 'error': 'No data available'}

        latest = df.iloc[-1]
        return {
            'symbol': symbol_ticker,
            'timestamp': str(df.index[-1]),
            'price': round(latest['close'], 2),
            'data_points': len(df),
            'trend': {
                'EMA7': round(latest.get('EMA7', 0), 2),
                'EMA14': round(latest.get('EMA14', 0), 2),
                'EMA50': round(latest.get('EMA50', 0), 2),
                'EMA200': round(latest.get('EMA200', 0), 2),
                'trend_signal': 'BULLISH' if latest.get('EMA7', 0) > latest.get('EMA50', 0) else 'BEARISH',
            },
            'momentum': {
                'RSI': round(latest.get('RSI', 0), 2),
                'RSI_signal': 'OVERBOUGHT' if latest.get('RSI', 50) > 70 else ('OVERSOLD' if latest.get('RSI', 50) < 30 else 'NEUTRAL'),
                'MACD_line': round(latest.get('MACD_line', 0), 4),
                'MACD_signal': round(latest.get('MACD_signal', 0), 4),
                'MACD_histogram': round(latest.get('MACD_diff', 0), 4),
                'MACD_crossover': 'BULLISH' if latest.get('MACD_diff', 0) > 0 else 'BEARISH',
            },
            'volatility': {
                'BB_upper': round(latest.get('BB_upper', 0), 2),
                'BB_mid': round(latest.get('BB_mid', 0), 2),
                'BB_lower': round(latest.get('BB_lower', 0), 2),
                'BB_position': 'ABOVE' if latest['close'] > latest.get('BB_upper', latest['close']) else (
                    'BELOW' if latest['close'] < latest.get('BB_lower', latest['close']) else 'INSIDE'
                ),
            },
            'volume': {
                'OBV': round(latest.get('OBV', 0), 0),
            },
        }
