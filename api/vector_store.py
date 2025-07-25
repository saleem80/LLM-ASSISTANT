import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, index_path="vector.index", metadata_path="metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(384)
            self.metadata = []

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add(self, texts, sources, embeddings):
        self.index.add(np.array(embeddings, dtype="float32"))
        for text, source in zip(texts, sources):
            self.metadata.append({"text": text, "source": source})

    def search(self, query_embedding, top_k=3):
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(np.array([query_embedding], dtype="float32"), top_k)
        return [self.metadata[idx] for idx in I[0] if idx < len(self.metadata)]


    def reset(self):
        self.index = faiss.IndexFlatL2(384)
        self.metadata = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
