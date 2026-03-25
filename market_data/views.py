"""API views for market data."""
import logging

from django.db.models import Count
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Symbol, OHLCV, PriceSnapshot
from .serializers import (
    SymbolSerializer, OHLCVSerializer, PriceSnapshotSerializer,
    IngestRequestSerializer,
)
from .services import MarketDataService, MarketDataAPIError

logger = logging.getLogger(__name__)


class SymbolViewSet(viewsets.ModelViewSet):
    """CRUD for stock symbols being tracked."""
    queryset = Symbol.objects.all()
    serializer_class = SymbolSerializer

    def get_queryset(self):
        qs = super().get_queryset().annotate(candle_count=Count('candles'))
        is_active = self.request.query_params.get('is_active')
        if is_active is not None:
            qs = qs.filter(is_active=is_active.lower() == 'true')
        return qs


class OHLCVViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only access to OHLCV candle data stored in DB."""
    serializer_class = OHLCVSerializer

    def get_queryset(self):
        qs = OHLCV.objects.select_related('symbol')
        symbol = self.request.query_params.get('symbol')
        if symbol:
            qs = qs.filter(symbol__ticker=symbol.upper())
        limit = self.request.query_params.get('limit')
        if limit:
            try:
                qs = qs[:int(limit)]
            except (ValueError, TypeError):
                pass
        return qs


class PriceView(APIView):
    """
    Get live prices from Market Data API.
    GET /api/market-data/prices/?symbols=AAPL,MSFT,GOOG
    """
    def get(self, request):
        symbols_param = request.query_params.get('symbols', '')
        if not symbols_param:
            return Response(
                {'error': 'Query param "symbols" is required (comma-separated)'},
                status=status.HTTP_400_BAD_REQUEST
            )

        symbols = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]
        service = MarketDataService()
        try:
            prices = service.get_prices_bulk(symbols)
            return Response({'s': 'ok', 'data': prices})
        except MarketDataAPIError as e:
            return Response({'s': 'error', 'errmsg': str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        finally:
            service.close()


class IngestView(APIView):
    """
    Trigger data ingestion from Market Data API.
    POST /api/market-data/ingest/
    Body: {"symbols": ["AAPL", "MSFT"], "resolution": "D", "countback": 365}
    """
    def post(self, request):
        serializer = IngestRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        symbols = serializer.validated_data['symbols']
        resolution = serializer.validated_data.get('resolution', 'D')
        countback = serializer.validated_data.get('countback', 365)

        service = MarketDataService()
        results = {'success': [], 'errors': []}

        try:
            for ticker in symbols:
                try:
                    count = service.sync_candles_to_db(ticker, resolution, countback)
                    service.sync_price_to_db(ticker)
                    results['success'].append({
                        'symbol': ticker,
                        'new_candles': count,
                    })
                except MarketDataAPIError as e:
                    logger.warning(f"Ingestion failed for {ticker}: {e}")
                    results['errors'].append({
                        'symbol': ticker,
                        'error': str(e),
                    })
        finally:
            service.close()

        return Response({
            's': 'ok',
            'ingested': len(results['success']),
            'failed': len(results['errors']),
            'details': results,
        })


class SnapshotListView(APIView):
    """Get all stored price snapshots."""
    def get(self, request):
        snapshots = PriceSnapshot.objects.select_related('symbol').all()
        serializer = PriceSnapshotSerializer(snapshots, many=True)
        return Response({'s': 'ok', 'data': serializer.data})
