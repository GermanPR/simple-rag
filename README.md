# Simple RAG System

A complete Retrieval-Augmented Generation (RAG) system built from scratch for PDF documents using FastAPI, Streamlit, and Mistral AI. This system implements hybrid search (TF-IDF + semantic), MMR diversification, and grounded answer generation with citations.

## 🌟 Features

- **📄 PDF Processing**: Extract text from PDF documents using pdfminer.six
- **🔍 Hybrid Search**: Combines TF-IDF keyword search with semantic similarity
- **🎯 Smart Retrieval**: MMR (Maximal Marginal Relevance) for result diversification
- **🤖 Grounded Generation**: Uses Mistral AI with strict citation requirements
- **⚡ FastAPI Backend**: RESTful API with `/ingest` and `/query` endpoints
- **🎨 Streamlit UI**: Interactive web interface with local or backend modes
- **💾 SQLite Storage**: Lightweight database with optimized vector storage
- **🛡️ Safety Features**: Evidence thresholds, hallucination filters, and disclaimers

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │    │   FastAPI API   │
│  (Local/Remote) │◄──►│   (/ingest,     │
│                 │    │    /query)      │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌──────────────────┐
            │  SQLite Database │
            │  ┌─────────────┐ │
            │  │ Documents   │ │
            │  │ Chunks      │ │
            │  │ Embeddings  │ │
            │  │ TF-IDF      │ │
            │  └─────────────┘ │
            └──────────────────┘
                     │
            ┌──────────────────┐
            │   Mistral AI     │
            │  ┌─────────────┐ │
            │  │ Embeddings  │ │
            │  │ Generation  │ │
            │  └─────────────┘ │
            └──────────────────┘
```

## 🔧 Pipeline Overview

### 1. **Ingestion Pipeline**
```
PDF → Text Extraction → Chunking → TF-IDF Indexing → Embedding → Storage
```

### 2. **Query Pipeline**
```
Query → Intent Detection → Rewriting → Hybrid Search → MMR → Generation → Citations
```

### 3. **Hybrid Retrieval**
- **TF-IDF**: `tf = 1 + log(count)`, `idf = log(N/(1+df))`
- **Semantic**: Cosine similarity on L2-normalized Mistral embeddings
- **Fusion**: `score = α×semantic + (1-α)×keyword` (default α=0.65)
- **MMR**: `λ×relevance - (1-λ)×max_similarity` for diversification

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to the project
cd simple-rag

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv add fastapi uvicorn requests numpy pdfminer.six pydantic streamlit python-multipart
uv add --dev ruff pytest basedpyright
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your Mistral API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_EMBED_MODEL=mistral-embed
MISTRAL_CHAT_MODEL=mistral-large-latest
DB_PATH=rag.sqlite3
```

### 3. Run the System

**Option A: Streamlit UI (Local Mode)**
```bash
uv run streamlit run ui/streamlit_app.py
```

**Option B: FastAPI Backend**
```bash
uv run uvicorn app.main:app --reload --port 8000
```

**Option C: Both (Backend + UI)**
```bash
# Terminal 1: Start backend
uv run uvicorn app.main:app --reload --port 8000

# Terminal 2: Set backend URL and start UI
export BACKEND_URL=http://localhost:8000
uv run streamlit run ui/streamlit_app.py
```

## 📚 Usage

### API Endpoints

**Ingest Documents** (`POST /ingest`)
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

Response:
```json
{
  "documents": [
    {"id": 1, "filename": "document1.pdf", "pages": 10}
  ],
  "chunks_indexed": 45,
  "success": true
}
```

**Query Documents** (`POST /query`)
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key benefits of machine learning?",
    "top_k": 5,
    "threshold": 0.18
  }'
```

Response:
```json
{
  "answer": "Machine learning offers several key benefits [document1.pdf p.3]: automated decision making, pattern recognition, and predictive analytics...",
  "citations": [
    {
      "chunk_id": 42,
      "filename": "document1.pdf",
      "page": 3,
      "snippet": "Machine learning algorithms can automate..."
    }
  ],
  "insufficient_evidence": false,
  "intent": "qa"
}
```

### Streamlit Interface

1. **Upload PDFs**: Use the sidebar file uploader
2. **Configure Parameters**: Adjust retrieval settings
3. **Ask Questions**: Type in the chat interface
4. **View Citations**: Expand citation panels for sources

## ⚙️ Configuration

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 8 | Final results to return |
| `rerank_k` | 20 | Candidates for reranking |
| `threshold` | 0.18 | Min semantic similarity |
| `alpha` | 0.65 | Semantic vs keyword weight |
| `lambda_param` | 0.7 | MMR relevance vs diversity |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | - | Your Mistral AI API key |
| `MISTRAL_EMBED_MODEL` | mistral-embed | Embedding model |
| `MISTRAL_CHAT_MODEL` | mistral-large-latest | Chat model |
| `DB_PATH` | rag.sqlite3 | Database file path |
| `CHUNK_SIZE` | 1800 | Target chunk size (chars) |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_chunker.py
uv run pytest tests/test_tfidf.py -v

# Run with coverage
uv run pytest --cov=app tests/
```

Test coverage includes:
- ✅ Chunking algorithms and overlap
- ✅ TF-IDF calculation and indexing
- ✅ Semantic search and cosine similarity
- ✅ Hybrid fusion and MMR diversification
- ✅ Database operations
- ✅ API endpoint functionality

## 📁 Project Structure

```
simple-rag/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   ├── ingest.py             # Document ingestion endpoint
│   │   └── query.py              # Query processing endpoint
│   ├── core/
│   │   ├── config.py             # Configuration settings
│   │   └── models.py             # Pydantic schemas
│   ├── ingest/
│   │   ├── pdf_extractor.py      # PDF text extraction
│   │   └── chunker.py            # Text chunking with overlap
│   ├── retriever/
│   │   ├── index.py              # SQLite database operations
│   │   ├── keyword.py            # TF-IDF search implementation
│   │   ├── semantic.py           # Cosine similarity search
│   │   └── fusion.py             # Hybrid fusion and MMR
│   ├── llm/
│   │   ├── mistral_client.py     # Mistral API client
│   │   └── prompts.py            # Prompt templates
│   └── logic/
│       ├── intent.py             # Query intent detection
│       └── postprocess.py        # Citation processing
├── ui/
│   └── streamlit_app.py          # Streamlit interface
├── tests/
│   ├── test_chunker.py           # Chunking tests
│   ├── test_tfidf.py             # TF-IDF tests
│   ├── test_semantic.py          # Semantic search tests
│   └── test_fusion.py            # Fusion and MMR tests
├── .env.example                  # Environment template
├── pyproject.toml               # UV project config
└── README.md                    # This file
```

## 🛡️ Safety & Guardrails

### Evidence Thresholds
- **Semantic Threshold**: Minimum cosine similarity (default: 0.18)
- **Insufficient Evidence**: Returns refusal when support is weak
- **Citation Requirements**: All claims must include `[filename p.X]` citations

### Intent Detection
- **Smalltalk**: Friendly responses for greetings/thanks
- **Structured**: Bullet points/lists for enumeration requests
- **Comparison**: Side-by-side analysis format
- **Q&A**: Standard question-answering

### Content Safety
- **PII Detection**: Warns about personal information
- **Medical/Legal**: Adds disclaimers for sensitive topics
- **Hallucination Filter**: Optional sentence-level verification

## 🎯 Design Decisions

### Why No External Vector DBs?
- **Simplicity**: SQLite BLOB storage is sufficient for most use cases
- **Portability**: Single-file database, easy deployment
- **Performance**: L2-normalized vectors + numpy for cosine similarity
- **Cost**: No additional infrastructure or API costs

### Why Hybrid Search?
- **Keyword Strengths**: Exact term matching, acronyms, proper nouns
- **Semantic Strengths**: Conceptual similarity, paraphrases, context
- **Best of Both**: α-weighted combination captures diverse relevance signals

### Why MMR Diversification?
- **Avoid Redundancy**: Prevents similar chunks from dominating results
- **Better Coverage**: Ensures diverse aspects of topics are represented
- **User Experience**: More informative and comprehensive answers

## 🚀 Production Deployment

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv venv && uv add fastapi uvicorn streamlit [other deps]

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Streamlit Deployment

For Streamlit Cloud deployment:

1. Create `.streamlit/secrets.toml`:
```toml
MISTRAL_API_KEY = "your_key_here"
DB_PATH = "rag.sqlite3"
```

2. Add to requirements:
```txt
fastapi
streamlit
pdfminer.six
numpy
requests
pydantic
```

### Environment Setup

**Production considerations**:
- Use environment-specific API keys
- Configure proper logging levels
- Set up health checks and monitoring
- Consider rate limiting for API endpoints
- Use proper CORS settings for web deployment

## 📊 Performance Tuning

### Database Optimizations
- **Indexes**: Created on frequently queried columns
- **WAL Mode**: Better concurrency for SQLite
- **Batch Operations**: Bulk embedding insertions
- **Connection Pooling**: Reuse database connections

### Search Optimizations
- **Embedding Cache**: In-memory storage of embeddings
- **Precomputed Norms**: L2-normalized storage
- **Token Filtering**: Stopword removal, minimum length
- **Result Limiting**: Configurable top-k and rerank-k

### Memory Usage
- **Streaming**: Process large PDFs in chunks
- **Lazy Loading**: Load embeddings on first search
- **Cache Management**: Clear caches when memory is tight
- **Batch Size**: Configurable batch sizes for embedding API calls

## 🤝 Contributing

1. **Code Style**: Use `ruff` for linting and formatting
2. **Testing**: Add tests for new features
3. **Documentation**: Update docstrings and README
4. **Type Hints**: Use proper Python type annotations

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run basedpyright
```

## 📄 License

This project is provided as-is for educational and research purposes. Please review the licenses of all dependencies:

- **FastAPI**: MIT License
- **Streamlit**: Apache 2.0 License
- **pdfminer.six**: MIT License
- **NumPy**: BSD License
- **Mistral AI**: API Terms of Service

## 🙏 Acknowledgments

Built following the detailed requirements in `instructions.md`. This implementation demonstrates:

- **No External Dependencies**: Custom TF-IDF, cosine similarity, and MMR implementations
- **Production Ready**: Comprehensive error handling, logging, and testing
- **Scalable Design**: Modular architecture with clear separation of concerns
- **Educational Value**: Well-documented code showing RAG system internals

**References:**
- [Mistral AI API Documentation](https://docs.mistral.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PDFMiner Documentation](https://pdfminersix.readthedocs.io/)

---

**Note**: Remember to set your `MISTRAL_API_KEY` environment variable before running the system!

For questions or issues, please refer to the inline documentation in the code or create an issue in the repository.
