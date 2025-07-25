from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore
from django.core.files.storage import default_storage
from django.conf import settings
from .llm import generate_answer
import os

from .cache import get_cached_answer, set_cached_answer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore()
store.load()

class AskQuestionView(APIView):
    def post(self, request):
        question = request.data.get("question", "").strip()
        if not question:
            return Response({"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST)

        query_embedding = embedding_model.encode(question).tolist()
        top_chunks = store.search(query_embedding, top_k=3)


        if not top_chunks:
            return Response({"answer": "No relevant information found.", "sources": []})

        # print("Top Chunks Used for Answer:")
        # for chunk in top_chunks:
        #     print(f"Source: {chunk['source']}")
        #     print(f"Text: {chunk['text'][:200]}...\n")

        context = "\n\n".join(chunk["text"] for chunk in top_chunks)
        # sources = list({chunk["source"] for chunk in top_chunks})
        sources = top_chunks[0]["source"] if top_chunks else None


        cached = get_cached_answer(context, question)
        if cached:
            return Response({
                "answer": cached["answer"], 
                "sources": sources,
                "cached": True
                })
        
        answer = generate_answer(context, question)

        set_cached_answer(context, question, {"answer": answer})

        return Response({
            "answer": answer,
            "sources": sources,
            "cached": False
        })



from .ingestion import ingest_pdf

class PDFIngestView(APIView):
    def post(self, request, format=None):
        # print("FILES:", request.FILES)
        # print("DATA:", request.data)
        file = request.FILES.get("document")
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        rel_path = os.path.join('files', file.name)
        save_path = os.path.join(settings.MEDIA_ROOT, rel_path)
        path = default_storage.save(save_path, file)

        try:
            ingest_pdf(path)
            return Response({"message": f"File '{file.name}' ingested successfully."})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
