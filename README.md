# ResearchVault: Offline NLP + RL Research Knowledge Agent

ResearchVault is an offline-first research intelligence application. 
It extracts and structures content from PDF research papers using NLP, builds persistent semantic/structured knowledge (Vector DB, Lexical DB, Knowledge Graph), and answers questions locally. 
A Reinforcement Learning (RL) agent dynamically selects the best retrieval strategy based on query context.

## Project Blueprint Modules
- `ingestion/`: PDF extraction (PyMuPDF4LLM)
- `nlp/`: Chunking, Section Classification, Entities (spaCy)
- `retrieval/`: FAISS, BM25, Knowledge Graph, Reranking
- `llm/`: Local Qwen-family GGUF Answer Generation
- `verification/`: Confidence, Abstention, Numeric Gates
- `rl/`: DQN Retrieval Strategy Selection
- `memory/`: SQLite Storage, Personalization
- `app/`: Streamlit UI & FastAPI
- `evaluation/`: Benchmark datasets and metrics

## Getting Started
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
