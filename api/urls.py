from django.urls import path, include
from.views import AskQuestionView, PDFIngestView


urlpatterns = [

    path('ask-question', AskQuestionView.as_view()),
    path("upload-pdf", PDFIngestView.as_view(), name="upload-pdf"),
]
