from django.urls import path
from .views import (
    chat, upload_pdf, home,
    get_sessions, get_messages, delete_session,
    login_page, signup_page,
    signup_user, login_user, logout_user
)

urlpatterns = [
    # 🔥 ROOT (IMPORTANT)
    path("", home),

    # pages
    path("signup-page/", signup_page),
    path("login-page/", login_page),

    # auth APIs
    path("signup/", signup_user),
    path("login/", login_user),
    path("logout/", logout_user),

    # chat system
    path("chat/", chat),
    path("upload_pdf/", upload_pdf),
    path("sessions/", get_sessions),

    # ❌ FIXED THESE TWO
    path("messages/<str:session_id>/", get_messages),
    path("delete-session/<str:session_id>/", delete_session),
]