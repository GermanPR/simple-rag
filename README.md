# Simple RAG System

A complete Retrieval-Augmented Generation (RAG) system built from scratch for PDF documents using FastAPI, Streamlit, and Mistral AI. This system implements hybrid search (TF-IDF + semantic), MMR diversification, and grounded answer generation with citations — **without external RAG libraries or vector databases**.

## Features

- **PDF Processing**: Extract text from PDF documents with page-level citations
- **Hybrid Search**: Combines TF-IDF keyword search with semantic similarity  
- **Smart Retrieval**: MMR (Maximal Marginal Relevance) for result diversification
- **Grounded Generation**: Uses Mistral AI with strict citation requirements
- **Dual Architecture**: FastAPI backend + Streamlit UI that can run standalone or connected
- **SQLite Storage**: Lightweight database with optimized vector storage
- **Safety Features**: Evidence thresholds, intent detection, and content filtering

## Architecture

The system has two main components:
1. **FastAPI Backend** (`src/app/`): RESTful API with `/ingest`, `/query`, `/collections` and `/collections/stats` endpoints  
2. **Streamlit UI** (`src/ui/`): Interactive web interface that can run standalone or connect to backend

### Core Pipeline
- **Ingestion**: PDF → text extraction → chunking → TF-IDF indexing → embedding → SQLite storage
- **Query**: Query → intent detection → hybrid search (TF-IDF + semantic) → MMR → generation → citations

## Quick Start

### 1. Setup with uv

```bash
# Clone and navigate to the project
cd simple-rag

# Create virtual environment and install dependencies
uv sync
```

### 2. Configure API Key

#### FastAPI

Copy `.env.example` to `.env` and add your Mistral API key:

```bash
cp .env.example .env
# Edit .env to add: MISTRAL_API_KEY=your_mistral_api_key_here
```

#### Streamlit UI

Copy `.env.example` to `.streamlit/secrets.toml` and add your Mistral API key:

```bash
mkdir .streamlit/
cp .env.example .streamlit/secrets.toml
# Edit .env to add: MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Run the System

**Option A: Streamlit UI (Local Mode)**
```bash
uv run poe streamlit-app # poe task defined in pyproject.toml
```

**Option B: FastAPI Backend + UI**
```bash
# Terminal 1: Start backend
uv run uvicorn src.app.main:app --reload --port 8000

# Terminal 2: Start UI with backend
BACKEND_URL=http://localhost:8000 uv run streamlit run src/ui/streamlit_app.py
```

## API Usage

Once the FastAPI backend is running on `http://localhost:8000`, you can interact with it using curl:

### Health Check
```bash
curl http://localhost:8000/health
```

### Ingest PDF Documents
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main benefits of machine learning?",
    "top_k": 5,
    "threshold": 0.4,
    "use_mmr": true
  }'
```

### Get Collection Statistics
```bash
curl http://localhost:8000/collections/stats
```

### Delete All Collections
```bash
curl -X DELETE http://localhost:8000/collections
```

## Dependencies

### Runtime Dependencies
- **fastapi**: Web framework for the REST API backend
- **uvicorn**: ASGI server for running FastAPI in production
- **streamlit**: Interactive web UI framework for the frontend
- **pydantic**: Data validation and settings management with type annotations
- **pydantic-settings**: Environment variable configuration management
- **mistralai**: Official Mistral AI client for embeddings and chat completions
- **numpy**: Numerical operations for embedding vectors and similarity calculations
- **aiosqlite**: Async SQLite adapter for database operations
- **httpx**: HTTP client for Streamlit to communicate with FastAPI backend
- **pymupdf**: PDF text extraction with page-level metadata (replaces pdfminer)
- **python-multipart**: FastAPI dependency for handling file uploads
- **typer**: Command-line interface framework with rich terminal output
- **rich**: Enhanced terminal formatting and progress displays

### Development Dependencies  
- **pytest**: Testing framework with async support and benchmarking
- **ruff**: Fast Python linter and formatter (replaces black, isort, flake8)
- **basedpyright**: Static type checker for Python
- **pre-commit**: Git hooks for automated code quality checks
- **poethepoet**: Task runner for development commands

## Configuration

The system uses environment variables defined in `src/app/core/config.py`:

### Required
- `MISTRAL_API_KEY`: Your Mistral AI API key

### Optional  
- `MISTRAL_EMBED_MODEL`: Embedding model (default: "mistral-embed")
- `MISTRAL_CHAT_MODEL`: Chat model (default: "mistral-large-latest")
- `DB_PATH`: SQLite database path (default: "rag.sqlite3")
- `BACKEND_URL`: For Streamlit to connect to FastAPI backend
- `CHUNK_SIZE`: Target chunk size in characters (default: 1800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## Project Structure

```
simple-rag/
├── src/
│   ├── app/                      # FastAPI backend
│   │   ├── main.py               # FastAPI application entry point
│   │   ├── api/                  # API endpoints
│   │   ├── core/                 # Configuration and models
│   │   ├── ingest/               # PDF processing and chunking
│   │   ├── retriever/            # Search implementations
│   │   ├── llm/                  # Mistral API client and prompts
│   │   ├── logic/                # Intent detection and postprocessing
│   │   ├── services/             # Business logic services
│   │   └── cli/                  # Command-line interface
│   ├── ui/                       # Streamlit interface
│   │   ├── streamlit_app.py      # Main UI application
│   │   ├── components/           # UI components
│   │   ├── handlers/             # Request handlers
│   │   └── utils/                # UI utilities
│   └── tests/                    # Test suite
├── pyproject.toml               # UV project configuration
├── CLAUDE.md                    # Development guidance
└── README.md                    # This file
```

## Key Implementation Details

### Custom Implementations (No External Libraries)
- **TF-IDF**: Custom tokenization, term frequency (`1 + log(count)`), document frequency indexing
- **Semantic Search**: L2-normalized Mistral embeddings with cosine similarity (dot product)  
- **Hybrid Fusion**: Alpha-weighted combination of semantic and keyword scores with min-max normalization
- **MMR Diversification**: Maximal Marginal Relevance to reduce redundancy in results

### Safety and Guardrails System
- **Evidence threshold refusal** (default: 0.4 similarity)
- **Citation requirements**: All answers include `[filename p.X]` citations
- **Intent detection**: LLM-based query analysis with fallback to pattern matching
- **Hallucination detection**: LLM-based verification that answers are grounded in context
- **PII detection**: LLM-based identification of personally identifiable information


## Development

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest src/tests/test_chunker.py -v
uv run pytest --cov=app src/tests/  # With coverage

# Run tests matching pattern
uv run pytest -k "test_tfidf"
```

### Code Quality
```bash
# Format and lint code
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues

# Type checking
uv run basedpyright
uv run basedpyright --level ERROR  # Show only errors
```

### CLI Commands
```bash
# View database statistics
uv run rag stats

# Debug queries with detailed retrieval info
uv run rag query "your question here" --k 10 --alpha 0.7

# Test LLM-based intent detection
uv run rag test-intent "What are the benefits of AI?"
```

### Database Schema (SQLite)
- `documents`: Document metadata
- `chunks`: Text chunks with position/page info  
- `embeddings`: Float32 embeddings stored as BLOB, L2-normalized
- `df`: Document frequency by chunk for TF-IDF
- `tokens`: Term frequency per chunk for TF-IDF

## References

- [Mistral AI API Documentation](https://docs.mistral.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
