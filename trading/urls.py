"""URL routing for trading app."""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'sessions', views.TrainingSessionViewSet, basename='training-session')

urlpatterns = [
    path('', include(router.urls)),
    path('train/', views.TrainView.as_view(), name='train-async'),
    path('train-sync/', views.TrainSyncView.as_view(), name='train-sync'),
]
