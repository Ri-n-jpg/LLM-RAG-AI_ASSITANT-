from django.urls import path
from .views import chat, upload_pdf, home,get_sessions,get_messages

urlpatterns = [
    path("", home),
    path("chat/", chat),
    path("upload_pdf/", upload_pdf),
path("sessions/", get_sessions),
path("messages/<str:session_id>/", get_messages),
]