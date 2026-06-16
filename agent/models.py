from django.db import models

class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
    ]
    role = models.CharField(max_length=20)  # user / assistant / system
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class Document(models.Model):
        title = models.CharField(max_length=255)
        file = models.FileField(upload_to="docs/")
        content = models.TextField(blank=True)
        created_at = models.DateTimeField(auto_now_add=True)

        def __str__(self):
            return self.title

class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    text = models.TextField()
    embedding = models.JSONField()  # store vector