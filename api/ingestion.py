import fitz 
import os
from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

store = VectorStore()
store.load()

def extract_chunks_from_pdf(path):
    doc = fitz.open(path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]

        for para in paragraphs:
            source = f"{os.path.basename(path)} - Page {page_num + 1}"
            chunks.append((para, source))

    return chunks

def embed_text(text):
    return embedding_model.encode(text).tolist()

def ingest_pdf(path):
    store.reset()
    chunks = extract_chunks_from_pdf(path)
    for text, source in chunks:
        embedding = embed_text(text)
        store.add([text], [source], [embedding])
    store.save()
