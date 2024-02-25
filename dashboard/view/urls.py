from django.urls import path, include
from . import views

urlpatterns = [
    path('',views.index, name = "index"),
    path('alerts/', views.alerts, name = "alerts"),
    path('nodes/', views.nodes, name="nodes"),
    path('terms/', views.terms, name="terms"),
    path('privacy/', views.privacy, name="privacy"),
    path('change_theme/', views.change_theme, name="change_theme"),

]