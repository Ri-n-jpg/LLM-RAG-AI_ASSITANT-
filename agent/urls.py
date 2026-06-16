from django.urls import path
from .views import chat
from .views import upload_pdf
from .views import split_text
urlpatterns = [
    path("chat/", chat),
    path("upload/", upload_pdf),
    path("split/", split_text)
]