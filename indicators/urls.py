"""URL routing for indicators app."""
from django.urls import path
from . import views

urlpatterns = [
    path('summary/', views.IndicatorSummaryView.as_view(), name='indicator-summary'),
    path('data/', views.IndicatorDataView.as_view(), name='indicator-data'),
]
