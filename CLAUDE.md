# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete Retrieval-Augmented Generation (RAG) system built from scratch for PDF documents. The system implements hybrid search (TF-IDF + semantic), MMR diversification, and grounded answer generation with citations using FastAPI, Streamlit, and Mistral AI - **without external RAG libraries or vector databases**.

## Architecture

The system has two main components:

1. **FastAPI Backend** (`app/`): RESTful API with `/ingest` and `/query` endpoints
2. **Streamlit UI** (`ui/`): Interactive web interface that can run standalone (local SQLite) or connect to the backend

### Core Pipeline
- **Ingestion**: PDF → text extraction → chunking → TF-IDF indexing → embedding → SQLite storage
- **Query**: Query → intent detection → hybrid search (TF-IDF + semantic) → MMR → generation → citations

### Key Components
- `app/api/`: FastAPI endpoints (`ingest.py`, `query.py`)
- `app/retriever/`: Search implementations (keyword, semantic, fusion with MMR)
- `app/ingest/`: PDF processing and text chunking
- `app/llm/`: Mistral API client and prompts
- `app/logic/`: Intent detection and postprocessing
- `app/core/`: Configuration, data models, monitoring, and interfaces
- `app/cli/`: Command-line interface for database inspection and debugging

## Development Commands

### Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (already defined in pyproject.toml)
uv install

# Install development dependencies
uv install --dev
```

### Running the System
```bash
# Start FastAPI backend
uv run uvicorn app.main:app --reload --port 8000

# Start Streamlit UI (local mode)
uv run streamlit run ui/streamlit_app.py

# Start Streamlit UI (backend mode)
export BACKEND_URL=http://localhost:8000
uv run streamlit run ui/streamlit_app.py

# Run both (separate terminals)
uv run uvicorn app.main:app --reload --port 8000
BACKEND_URL=http://localhost:8000 uv run streamlit run ui/streamlit_app.py
```

### Development Tools
```bash
# Code formatting and linting
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues

# Type checking
uv run basedpyright
uv run basedpyright --level ERROR  # Show only errors

# Run tests
uv run pytest
uv run pytest src/tests/test_chunker.py -v  # Single test file
uv run pytest --cov=app src/tests/  # With coverage
uv run pytest -k "test_tfidf" # Run tests matching pattern
```

### CLI Commands
The system includes a CLI for database inspection and debugging:

```bash
# View database statistics
uv run rag stats

# Inspect chunks in the database
uv run rag chunks --limit 20 --content
uv run rag chunks --doc "filename.pdf" --page 5

# Debug queries with detailed retrieval info
uv run rag query "your question here" --k 10 --alpha 0.7
uv run rag query "your question here" --no-llm  # Skip LLM, show only retrieval

# Test LLM-based intent detection and query rewriting
uv run rag test-intent "What are the benefits of AI?"
uv run rag test-intent "List the steps to implement ML" --simple
uv run rag test-intent "Compare Python vs Java" --history "user:What should I learn?" --history "assistant:Both are great choices"
uv run rag test-intent "Summarize key points" --fallback  # Use pattern matching only

# Alternative usage (via convenience scripts)
python rag_cli.py stats
python rag_cli.py chunks --content
python test_intent.py "Your query here" --simple
```

### Pre-commit Hooks
The project uses pre-commit hooks to automatically check code quality on every commit:

```bash
# Install pre-commit hooks (one-time setup)
uv run pre-commit install

# Run pre-commit checks on all files manually
uv run pre-commit run --all-files

# Update pre-commit hook versions
uv run pre-commit autoupdate
```

**Automated checks on commit:**
- **ruff format**: Auto-formats Python code
- **ruff check**: Lints code and fixes auto-fixable issues
- **basedpyright**: Type checking (ERROR level only, warnings ignored)
- **Basic file checks**: Trailing whitespace, end-of-file, merge conflicts
- **Security checks**: Bandit security scanner (with project-appropriate exclusions)

## Configuration

The system uses environment variables defined in `app/core/config.py`:

### Required
- `MISTRAL_API_KEY`: Your Mistral AI API key

### Optional
- `MISTRAL_EMBED_MODEL`: Embedding model (default: "mistral-embed")
- `MISTRAL_CHAT_MODEL`: Chat model (default: "mistral-large-latest")
- `DB_PATH`: SQLite database path (default: "rag.sqlite3")
- `BACKEND_URL`: For Streamlit to connect to FastAPI backend
- `CHUNK_SIZE`: Target chunk size in characters (default: 1800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

Create `.env` file from `.env.example` for local development.

## Key Implementation Details

### Core Architecture Pattern
The system follows a **service-oriented architecture** with clear separation:
- **RAGQueryService** (`app/services/query_service.py`): Unified service encapsulating the entire RAG pipeline, used by both sync (CLI/UI) and async (API) interfaces
- **Layered Components**: API endpoints delegate to services, services coordinate retrieval and generation, core components handle specific algorithms
- **Database Abstraction**: Both sync (`DatabaseManager`) and async (`AsyncDatabaseManager`) versions for different contexts

### Custom Implementations (No External Libraries)
- **TF-IDF**: Custom tokenization, term frequency (`1 + log(count)`), document frequency indexing
- **Semantic Search**: L2-normalized Mistral embeddings with cosine similarity (dot product)
- **Hybrid Fusion**: Alpha-weighted combination of semantic and keyword scores with min-max normalization
- **MMR Diversification**: Maximal Marginal Relevance to reduce redundancy in results

### Database Schema (SQLite)
- `documents`: Document metadata
- `chunks`: Text chunks with position/page info
- `embeddings`: Float32 embeddings stored as BLOB, L2-normalized
- `df`: Document frequency by chunk for TF-IDF
- `tokens`: Term frequency per chunk for TF-IDF

### Safety and Guardrails System
The system implements comprehensive safety checks in the postprocessing pipeline:
- **Evidence threshold refusal** (default: 0.18 similarity)
- **Citation requirements**: All answers include `[filename p.X]` citations
- **Intent detection**: LLM-based query analysis with fallback to pattern matching
- **Hallucination detection**: LLM-based verification that answers are grounded in context
- **PII detection**: LLM-based identification of personally identifiable information
- **Fail-safe behavior**: Defaults to blocking content when safety checks fail
- **Standardized error messages**: Clear user-facing messages for blocked content

### Performance Monitoring
- Built-in performance monitoring with `@performance_monitor` decorator
- Metrics collection for execution time and optional memory usage
- Automatic logging of slow operations (>1 second)
- Context manager for timing arbitrary code blocks

## Testing

The test suite covers core algorithms:
- `test_chunker.py`: Text chunking with overlap
- `test_tfidf.py`: TF-IDF calculation and scoring
- `test_semantic.py`: Cosine similarity and ranking
- `test_fusion.py`: Hybrid fusion and MMR behavior
- `test_config.py`: Configuration validation
- `test_monitoring.py`: Performance monitoring functionality

Tests are located in `src/tests/` and use pytest with fixtures for database setup.

## API Endpoints

- `POST /ingest`: Upload PDF files for indexing
- `POST /query`: Ask questions and get grounded answers with citations
- `GET /health`: System health check
- `GET /`: API information

## Important Implementation Notes

### Code Organization Principles
- **Async/Sync Duality**: Many components have both sync and async versions (e.g., `DatabaseManager`/`AsyncDatabaseManager`) to support different usage contexts
- **Interface Segregation**: Core interfaces defined in `app/core/interfaces.py` allow for clean testing and future extensibility
- **Pydantic Models**: All API schemas and internal data structures use Pydantic for validation (`app/core/models.py`)
- **Global State Management**: Singleton pattern for shared resources like Mistral client and database connections

### Data Flow Architecture
- **Ingestion Flow**: `PDF → Chunker → DatabaseManager → EmbeddingManager → SQLite`
- **Query Flow**: `Query → RAGQueryService → HybridRetriever → AnswerPostProcessor → Response`
- **Safety Pipeline**: Every generated answer passes through hallucination and PII detection before user delivery

### Key Technical Decisions
- **SQLite as Vector Store**: Embeddings stored as float32 BLOBs for simplicity and portability
- **Custom Algorithm Implementations**: All retrieval algorithms built from scratch without external RAG libraries
- **LLM-Powered Safety**: Uses Mistral API for both content generation and safety verification
- **Fail-Safe Design**: Safety checks default to blocking content when API calls fail, ensuring secure operation