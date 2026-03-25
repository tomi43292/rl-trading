"""API views for technical indicators."""
import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .services import IndicatorService
from .serializers import IndicatorRequestSerializer

logger = logging.getLogger(__name__)


class IndicatorSummaryView(APIView):
    """
    Get latest indicator summary for a symbol.
    GET /api/indicators/summary/?symbol=AAPL
    """
    def get(self, request):
        symbol = request.query_params.get('symbol')
        if not symbol:
            return Response(
                {'error': 'Query param "symbol" is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            summary = IndicatorService.get_indicator_summary(symbol.upper())
            return Response({'s': 'ok', 'data': summary})
        except ValueError as e:
            return Response({'s': 'error', 'errmsg': str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception(f"Indicator calculation failed for {symbol}")
            return Response({'s': 'error', 'errmsg': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class IndicatorDataView(APIView):
    """
    Get full indicator dataset for a symbol (used by the RL agent).
    GET /api/indicators/data/?symbol=AAPL&limit=500
    Returns JSON array of all OHLCV + indicator values.
    """
    def get(self, request):
        serializer = IndicatorRequestSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        symbol = serializer.validated_data['symbol']
        limit = serializer.validated_data.get('limit', 500)

        try:
            df = IndicatorService.calculate_all(symbol, limit=limit)
            records = df.reset_index().to_dict(orient='records')

            # Convert timestamps to ISO strings
            for record in records:
                record['timestamp'] = record['timestamp'].isoformat()

            return Response({
                's': 'ok',
                'symbol': symbol,
                'data_points': len(records),
                'columns': list(df.columns),
                'data': records,
            })
        except ValueError as e:
            return Response({'s': 'error', 'errmsg': str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception(f"Indicator data failed for {symbol}")
            return Response({'s': 'error', 'errmsg': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
