# ml_app/urls.py
from django.urls import path
from . import views

app_name = "ml_app"

urlpatterns = [
    path("", views.index, name="home"),
    path("predict/", views.predict_page, name="predict"),
    path("about/", views.about, name="about"),
    path("debug-thr/", views.debug_thr, name="debug-thr"),
]
