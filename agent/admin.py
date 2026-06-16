from django.contrib import admin
from .models import ChatMessage, Document
admin.site.register(ChatMessage)
admin.site.register(Document)