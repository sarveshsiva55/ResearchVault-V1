import streamlit as st
import os
import sys
import tempfile
import time

# Add the project root to the python path so it can find the other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import time
from config.settings import DB_PATH
from ingestion.pdf_parser import PDFIngestor
from nlp.chunker import TextChunker
from nlp.embedder import DocumentEmbedder
from retrieval.faiss_store import FAISSRetriever
from retrieval.bm25_store import BM25Retriever
from memory.db import DatabaseManager
from llm.generator import LocalLLMGenerator

st.set_page_config(page_title="ResearchVault", layout="wide")

# =====================================================================
# EFFICIENCY: Caching Heavy Models and Connections (Section 14)
# =====================================================================
@st.cache_resource
def load_db():
    return DatabaseManager(DB_PATH)

@st.cache_resource
def load_chunker():
    return TextChunker()

@st.cache_resource
def load_embedder():
    return DocumentEmbedder()

@st.cache_resource
def load_faiss():
    return FAISSRetriever()

@st.cache_resource
def load_bm25():
    return BM25Retriever()

@st.cache_resource
def load_llm():
    return LocalLLMGenerator()

# Load all resources once per app lifecycle
db = load_db()
chunker = load_chunker()
embedder = load_embedder()
faiss_store = load_faiss()
bm25_store = load_bm25()
llm_gen = load_llm()

# =====================================================================
# UI / UX Design (Section 18)
# =====================================================================
st.title("ResearchVault")
st.subheader("Offline NLP + RL Research Knowledge Agent")

# Sidebar for corpus upload and management
with st.sidebar:
    st.header("Corpus Management")
    uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            ingestor = PDFIngestor()
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save to temp file for PyMuPDF4LLM
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # 1. Parse PDF
                parsed_data = ingestor.process_pdf(tmp_path)
                file_hash = parsed_data["file_hash"]
                
                # 2. Add to DB and check for duplicate
                cursor = db.conn.cursor()
                cursor.execute("SELECT paper_id FROM papers WHERE file_hash = ?", (file_hash,))
                existing_paper = cursor.fetchone()
                
                if existing_paper:
                    status_text.text(f"Skipping {uploaded_file.name} (already indexed).")
                    os.remove(tmp_path)
                    progress_bar.progress((i + 1) / total_files)
                    continue
                
                cursor.execute("INSERT INTO papers (file_hash, title, path, page_count) VALUES (?, ?, ?, ?)", 
                               (file_hash, uploaded_file.name, tmp_path, len(parsed_data["pages"])))
                db.conn.commit()
                
                # 3. Chunk and Embed
                chunks_to_embed = []
                for page_dict in parsed_data["pages"]:
                    page_text = page_dict.get("text", "")
                    page_chunks = chunker.chunk_page(page_text)
                    chunks_to_embed.extend(page_chunks)
                
                if chunks_to_embed:
                    # Embed in batches
                    status_text.text(f"Embedding {len(chunks_to_embed)} chunks for {uploaded_file.name}...")
                    embeddings = embedder.embed_texts(chunks_to_embed)
                    faiss_store.add_embeddings(embeddings)
                    bm25_store.add_texts(chunks_to_embed)
                    
                    # Optional: Store chunks in DB...
                
                # Cleanup
                os.remove(tmp_path)
                progress_bar.progress((i + 1) / total_files)
                
            # Save indexes
            faiss_store.save()
            bm25_store.save()
            
            status_text.text("Processing complete.")
            st.success("Corpus updated successfully.")
        else:
            st.warning("Please upload a PDF first.")
            
    st.markdown("---")
    
    # Query basic stats
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    paper_count = cursor.fetchone()[0]
    
    st.metric("Papers in Corpus", paper_count)
    st.metric("Indexed Chunks", len(bm25_store.corpus))
    
    st.markdown("---")
    st.text("Offline Status: Local Engine ON 🟢")

# Main QA interface
st.header("Ask your research knowledge base...")
query = st.text_input("Enter your research question:")

if st.button("Search"):
    if query:
        start_time = time.time()
        with st.spinner("Retrieving evidence..."):
            # 1. Retrieve Dense
            q_emb = embedder.embed_query(query)
            d_scores, d_indices = faiss_store.search(q_emb, top_k=6)
            
            # 2. Retrieve BM25
            b_scores, b_indices = bm25_store.search(query, top_k=6)
            
            # (In the final pipeline, this is where RRF and RL policy steps would run)
            # For efficiency now, we deduplicate indices retrieved
            retrieved_chunks = []
            
            # Mock retrieving from DB using BM25 cache for now
            for idx in set(list(d_indices) + list(b_indices)):
                if idx < len(bm25_store.corpus):
                    retrieved_chunks.append(bm25_store.corpus[idx])
            
            # Bound evidence
            final_evidence = retrieved_chunks[:6]
        
        with st.spinner("Generating grounded answer..."):
            answer = llm_gen.generate_answer(query, final_evidence)
            
        latency = time.time() - start_time
        
        st.markdown("### Answer")
        st.write(answer)
        st.caption(f"Retrieved and generated in {latency:.2f} seconds.")
        
        st.markdown("### Evidence Used")
        for i, ev in enumerate(final_evidence):
            with st.expander(f"Evidence Snippet {i+1}"):
                st.write(ev)
        
        st.markdown("### Feedback")
        col1, col2, col3 = st.columns([1,1,4])
        with col1:
            st.button("👍 Correct")
        with col2:
            st.button("👎 Incorrect")
        with col3:
            st.button("📝 Correct the answer")
