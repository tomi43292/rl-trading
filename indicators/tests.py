"""Tests for indicators app."""
from decimal import Decimal
from datetime import datetime, timedelta
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APITestCase
from rest_framework import status

from market_data.models import Symbol, OHLCV
from .services import IndicatorService


class IndicatorServiceTest(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(ticker='AAPL', name='Apple Inc.')
        base_time = timezone.now() - timedelta(days=300)
        # Create enough candles for indicators to work (need ~200 minimum for EMA200)
        for i in range(250):
            price = Decimal('150') + Decimal(str(i * 0.1))
            OHLCV.objects.create(
                symbol=self.symbol,
                timestamp=base_time + timedelta(days=i),
                open=price - Decimal('1'),
                high=price + Decimal('2'),
                low=price - Decimal('2'),
                close=price,
                volume=1000000 + i * 1000,
            )

    def test_get_dataframe(self):
        df = IndicatorService.get_dataframe('AAPL')
        self.assertEqual(len(df), 250)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)

    def test_get_dataframe_unknown_symbol(self):
        with self.assertRaises(ValueError):
            IndicatorService.get_dataframe('INVALID')

    def test_calculate_all_indicators(self):
        df = IndicatorService.calculate_all('AAPL')
        self.assertGreater(len(df), 0)
        expected_columns = ['EMA7', 'EMA14', 'EMA50', 'EMA200', 'RSI', 'MACD_line', 'OBV']
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_indicator_summary(self):
        summary = IndicatorService.get_indicator_summary('AAPL')
        self.assertEqual(summary['symbol'], 'AAPL')
        self.assertIn('trend', summary)
        self.assertIn('momentum', summary)
        self.assertIn('volatility', summary)
        self.assertIn('volume', summary)
        self.assertIn('trend_signal', summary['trend'])
        self.assertIn('RSI_signal', summary['momentum'])


class IndicatorAPITest(APITestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(ticker='AAPL', name='Apple Inc.')
        base_time = timezone.now() - timedelta(days=300)
        for i in range(250):
            price = Decimal('150') + Decimal(str(i * 0.1))
            OHLCV.objects.create(
                symbol=self.symbol,
                timestamp=base_time + timedelta(days=i),
                open=price - Decimal('1'),
                high=price + Decimal('2'),
                low=price - Decimal('2'),
                close=price,
                volume=1000000 + i * 1000,
            )

    def test_summary_missing_symbol(self):
        response = self.client.get('/api/indicators/summary/')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_summary_unknown_symbol(self):
        response = self.client.get('/api/indicators/summary/?symbol=INVALID')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_summary_success(self):
        response = self.client.get('/api/indicators/summary/?symbol=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['s'], 'ok')
        self.assertIn('trend', response.data['data'])

    def test_data_endpoint(self):
        response = self.client.get('/api/indicators/data/?symbol=AAPL&limit=250')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['symbol'], 'AAPL')
        self.assertGreater(response.data['data_points'], 0)
