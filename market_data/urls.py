"""URL routing for market_data app."""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'symbols', views.SymbolViewSet, basename='symbol')
router.register(r'candles', views.OHLCVViewSet, basename='candle')

urlpatterns = [
    path('', include(router.urls)),
    path('prices/', views.PriceView.as_view(), name='live-prices'),
    path('ingest/', views.IngestView.as_view(), name='ingest'),
    path('snapshots/', views.SnapshotListView.as_view(), name='snapshots'),
]
