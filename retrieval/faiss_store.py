import faiss
import numpy as np
import os
from config.settings import VECTOR_INDEX_PATH

class FAISSRetriever:
    def __init__(self, dim: int = 384, index_path: str = VECTOR_INDEX_PATH):
        self.dim = dim
        self.index_path = index_path
        
        if os.path.exists(self.index_path):
            print(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS IndexFlatIP")
            # Inner product index for normalized embeddings (Cosine Similarity)
            self.index = faiss.IndexFlatIP(self.dim)
            
    def add_embeddings(self, embeddings: np.ndarray):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(np.float32(embeddings))
        
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 12):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(np.float32(query_embedding), top_k)
        return distances[0], indices[0]

if __name__ == "__main__":
    retriever = FAISSRetriever()
    print("FAISS Retriever Ready.")
