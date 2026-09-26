import os
import pymupdf4llm
import hashlib
from typing import Dict, Any

class PDFIngestor:
    def __init__(self, cache_dir: str = "data/cached_pdfs"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _compute_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a PDF using PyMuPDF4LLM.
        Extracts markdown with page-aware metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_hash = self._compute_hash(file_path)
        
        # In a full implementation, we'd check the SQLite DB or cache here 
        # to avoid re-parsing identical documents (Section 14).
        
        # PyMuPDF4LLM natively extracts structure, tables, and page metadata
        print(f"Extracting markdown from {file_path} using PyMuPDF4LLM...")
        md_text = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        
        # md_text is a list of dictionaries, one per page if page_chunks=True
        return {
            "file_path": file_path,
            "file_hash": file_hash,
            "pages": md_text
        }

if __name__ == "__main__":
    # Test script placeholder
    print("PDF Ingestor Module Ready.")
