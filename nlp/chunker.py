import spacy
from typing import List
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class TextChunker:
    def __init__(self):
        # We use a lightweight spaCy model for sentence boundary detection.
        # DISABLING ner and parser makes it MUCH faster and more efficient!
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer", "textcat"])
            self.nlp.add_pipe("sentencizer")
        except OSError:
            print("Downloading en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer", "textcat"])
            self.nlp.add_pipe("sentencizer")
        
    def chunk_page(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Splits text into semantic chunks trying to respect sentence boundaries.
        Efficient token approximation by whitespace splitting.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Determine overlap
                overlap_chunk = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s.split()) <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_length += len(s.split())
                    else:
                        break
                        
                current_chunk = overlap_chunk
                current_length = overlap_length
                
            current_chunk.append(sentence)
            current_length += sentence_length
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

if __name__ == "__main__":
    chunker = TextChunker()
    print("Chunker Ready (Efficient version).")
