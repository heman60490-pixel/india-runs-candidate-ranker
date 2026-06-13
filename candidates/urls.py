from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),        # homepage
    path('rank/', views.get_rankings, name='rank'),  # ranking page
]