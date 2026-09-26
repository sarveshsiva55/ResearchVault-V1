from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class DocumentEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def embed_texts(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """
        Generates dense embeddings for a list of strings.
        Returns a numpy array of shape (len(texts), hidden_dim).
        """
        # Prefixing recommended by BGE for passages if doing retrieval
        # Note: queries usually need a specific prefix like "Represent this sentence for searching relevant passages: "
        embeddings = self.model.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
        return embeddings

    def embed_query(self, query: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Embeds a search query with the BGE specific instruction prefix.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_query = instruction + query
        embedding = self.model.encode([formatted_query], normalize_embeddings=normalize_embeddings)
        return embedding[0]

if __name__ == "__main__":
    embedder = DocumentEmbedder()
    print("Embedder Ready.")
