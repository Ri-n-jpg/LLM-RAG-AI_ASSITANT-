from django.urls import path
from .views import chat, upload_pdf, home

urlpatterns = [
    path("", home),
    path("chat/", chat),
    path("upload_pdf/", upload_pdf),
]