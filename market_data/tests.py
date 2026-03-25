"""Tests for market_data app."""
from decimal import Decimal
from unittest.mock import patch, MagicMock

from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status

from .models import Symbol, OHLCV, PriceSnapshot
from .services import MarketDataService, MarketDataAPIError


class SymbolModelTest(TestCase):
    def test_create_symbol(self):
        symbol = Symbol.objects.create(ticker='AAPL', name='Apple Inc.')
        self.assertEqual(str(symbol), 'AAPL')
        self.assertTrue(symbol.is_active)

    def test_ticker_unique(self):
        Symbol.objects.create(ticker='AAPL')
        with self.assertRaises(Exception):
            Symbol.objects.create(ticker='AAPL')


class SymbolAPITest(APITestCase):
    def test_list_symbols(self):
        Symbol.objects.create(ticker='AAPL', name='Apple')
        Symbol.objects.create(ticker='MSFT', name='Microsoft')
        response = self.client.get('/api/market-data/symbols/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_create_symbol(self):
        response = self.client.post('/api/market-data/symbols/', {
            'ticker': 'AAPL',
            'name': 'Apple Inc.',
        })
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['ticker'], 'AAPL')

    def test_create_symbol_uppercase(self):
        response = self.client.post('/api/market-data/symbols/', {
            'ticker': 'aapl',
        })
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['ticker'], 'AAPL')


class PriceViewTest(APITestCase):
    def test_prices_missing_symbols_param(self):
        response = self.client.get('/api/market-data/prices/')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('market_data.views.MarketDataService')
    def test_prices_success(self, MockService):
        mock_instance = MockService.return_value
        mock_instance.get_prices_bulk.return_value = [
            {'symbol': 'AAPL', 'price': 150.50},
        ]
        response = self.client.get('/api/market-data/prices/?symbols=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['s'], 'ok')


class IngestViewTest(APITestCase):
    @patch('market_data.views.MarketDataService')
    def test_ingest_success(self, MockService):
        mock_instance = MockService.return_value
        mock_instance.sync_candles_to_db.return_value = 100
        mock_instance.sync_price_to_db.return_value = MagicMock()

        response = self.client.post('/api/market-data/ingest/', {
            'symbols': ['AAPL'],
            'resolution': 'D',
            'countback': 30,
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['ingested'], 1)

    def test_ingest_empty_symbols(self):
        response = self.client.post('/api/market-data/ingest/', {
            'symbols': [],
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class MarketDataServiceTest(TestCase):
    @patch('market_data.services.httpx.Client')
    def test_get_price_caches_result(self, MockClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            's': 'ok',
            'mid': [150.50],
            'change': [1.25],
            'changepct': [0.84],
            'volume': [1000000],
            'updated': [1700000000],
        }
        MockClient.return_value.get.return_value = mock_response

        service = MarketDataService()
        service._client = MockClient.return_value

        result = service.get_price('AAPL')
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['price'], 150.50)
