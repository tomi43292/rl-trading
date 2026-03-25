"""API views for the trading app."""
import logging

from django.db.models import Count
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import TrainingSession, Trade
from .serializers import (
    TrainingSessionSerializer, TrainingSessionListSerializer,
    TradeSerializer, TrainRequestSerializer,
)
from .services import TradingService
from .tasks import train_agent_async

logger = logging.getLogger(__name__)


class TrainingSessionViewSet(viewsets.ReadOnlyModelViewSet):
    """
    List and retrieve training sessions.
    POST /api/trading/sessions/train/ to start a new training.
    """
    queryset = TrainingSession.objects.select_related('symbol').all()

    def get_serializer_class(self):
        if self.action == 'list':
            return TrainingSessionListSerializer
        return TrainingSessionSerializer

    def get_queryset(self):
        qs = super().get_queryset().annotate(trades_count=Count('trades'))
        symbol = self.request.query_params.get('symbol')
        if symbol:
            qs = qs.filter(symbol__ticker=symbol.upper())
        status_filter = self.request.query_params.get('status')
        if status_filter:
            qs = qs.filter(status=status_filter.upper())
        return qs

    @action(detail=True, methods=['get'])
    def summary(self, request, pk=None):
        """Get detailed summary of a training session."""
        try:
            summary = TradingService.get_session_summary(int(pk))
            return Response({'s': 'ok', 'data': summary})
        except TrainingSession.DoesNotExist:
            return Response(
                {'s': 'error', 'errmsg': 'Session not found'},
                status=status.HTTP_404_NOT_FOUND
            )

    @action(detail=True, methods=['get'])
    def trades(self, request, pk=None):
        """Get all trades for a specific training session."""
        trades = Trade.objects.filter(session_id=pk).select_related('symbol')
        serializer = TradeSerializer(trades, many=True)
        return Response({'s': 'ok', 'data': serializer.data})


class TrainView(APIView):
    """
    Start a new RL agent training session.
    POST /api/trading/train/
    Body: {"symbol": "AAPL", "episodes": 100, "batch_size": 32, "initial_cash": 10000}

    Training runs asynchronously via Celery.
    Returns the session ID to track progress.
    """
    def post(self, request):
        serializer = TrainRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        symbol = data['symbol']
        episodes = data.get('episodes', 100)
        batch_size = data.get('batch_size', 32)
        initial_cash = data.get('initial_cash', 10000)

        # Dispatch to Celery for async execution
        task = train_agent_async.delay(symbol, episodes, batch_size, initial_cash)

        return Response({
            's': 'ok',
            'message': f'Training started for {symbol}',
            'task_id': task.id,
            'symbol': symbol,
            'episodes': episodes,
            'note': 'Training runs asynchronously. Check /api/trading/sessions/ for results.',
        }, status=status.HTTP_202_ACCEPTED)


class TrainSyncView(APIView):
    """
    Train synchronously (for development/testing, blocks until done).
    POST /api/trading/train-sync/
    """
    def post(self, request):
        serializer = TrainRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data

        try:
            session = TradingService.train_agent(
                symbol_ticker=data['symbol'],
                episodes=data.get('episodes', 100),
                batch_size=data.get('batch_size', 32),
                initial_cash=data.get('initial_cash', 10000),
            )

            return Response({
                's': 'ok',
                'session_id': session.id,
                'symbol': session.symbol.ticker,
                'status': session.status,
                'profit_loss': str(session.profit_loss),
                'profit_loss_pct': session.profit_loss_pct,
                'final_portfolio_value': str(session.final_portfolio_value),
            })
        except ValueError as e:
            return Response(
                {'s': 'error', 'errmsg': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return Response(
                {'s': 'error', 'errmsg': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
