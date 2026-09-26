from rank_bm25 import BM25Okapi
import json
import os

class BM25Retriever:
    def __init__(self, cache_path: str = "data/bm25_corpus.json"):
        self.cache_path = cache_path
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        
        self.load()

    def _tokenize(self, text: str):
        return text.lower().split()

    def add_texts(self, texts: list[str]):
        self.corpus.extend(texts)
        self.tokenized_corpus.extend([self._tokenize(t) for t in texts])
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1.2, b=0.75) # defaults from Section 7.4

    def search(self, query: str, top_k: int = 12):
        if not self.bm25:
            return [], []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = [self.corpus[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        return top_scores, top_indices

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.corpus, f)

    def load(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                texts = json.load(f)
                self.add_texts(texts)

if __name__ == "__main__":
    retriever = BM25Retriever()
    print("BM25 Retriever Ready.")
