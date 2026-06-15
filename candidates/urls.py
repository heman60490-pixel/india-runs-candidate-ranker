from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('rank/', views.get_rankings, name='rank'),
    path('download/', views.download_results, name='download'),
]