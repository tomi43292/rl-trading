"""URL configuration for RL Trading project."""
from django.contrib import admin
from django.urls import path, include
from rest_framework.response import Response
from rest_framework.decorators import api_view


@api_view(['GET'])
def api_root(request):
    return Response({
        'project': 'RL Trading System',
        'description': 'Reinforcement Learning for stock trading using Market Data API',
        'endpoints': {
            'market_data': '/api/market-data/',
            'indicators': '/api/indicators/',
            'trading': '/api/trading/',
            'admin': '/admin/',
        },
        'stack': ['Django', 'DRF', 'Redis', 'Celery', 'Pandas', 'TensorFlow', 'Docker'],
    })


urlpatterns = [
    path('', api_root, name='api-root'),
    path('admin/', admin.site.urls),
    path('api/market-data/', include('market_data.urls')),
    path('api/indicators/', include('indicators.urls')),
    path('api/trading/', include('trading.urls')),
]
