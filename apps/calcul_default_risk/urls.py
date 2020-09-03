from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    # path("api/", views.---as_view(), name="create"),
]