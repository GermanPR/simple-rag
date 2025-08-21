"""Streamlit UI for the Simple RAG system."""

import asyncio
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# Constants
HTTP_OK = 200
HTTP_NOT_FOUND = 404

# Add app directory to path so we can import our modules
app_root = Path(__file__).parent.parent
sys.path.insert(0, str(app_root))

# ruff: noqa: E402
from app.core.config import config
from app.core.logging_config import get_logger
from app.core.logging_config import setup_logging
from app.ingest.chunker import chunk_text
from app.ingest.pdf_extractor import extract_pdf_text
from app.llm.mistral_client import get_embedding_manager
from app.llm.mistral_client import get_mistral_client
from app.logic.intent import ConversationMessage
from app.retriever.index import DatabaseManager
from app.retriever.keyword import KeywordSearcher
from app.retriever.semantic import SemanticSearcher
from app.services.query_service import RAGQueryService

# Configure logging
setup_logging()
logger = get_logger(__name__.split(".")[-1])

# Page config
st.set_page_config(
    page_title="Simple RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    margin-bottom: 1rem;
}
.citation-box {
    background-color: #f0f2f6;
    border-left: 4px solid #1f77b4;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 4px;
}
.insufficient-evidence {
    color: #ff6b6b;
    font-weight: bold;
}
.success-message {
    color: #51cf66;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitRAG:
    """Main class for Streamlit RAG application."""

    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL")
        self.use_backend = bool(self.backend_url)

        # Initialize instance variables
        self.db_manager = None
        self.keyword_searcher = None
        self.semantic_searcher = None
        self.mistral_client = None
        self.embedding_manager = None
        self.query_service = None

        # Initialize session state
        self._init_session_state()

        # Initialize local components if not using backend
        if not self.use_backend:
            self._init_local_components()

    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "db_initialized" not in st.session_state:
            st.session_state.db_initialized = False

    def _init_local_components(self):
        """Initialize local RAG components when not using backend API."""
        try:
            # Get database path from config or Streamlit secrets
            db_path = st.secrets.get("DB_PATH", config.DB_PATH)

            # Initialize database
            if not st.session_state.db_initialized:
                self.db_manager = DatabaseManager(db_path)
                st.session_state.db_initialized = True
            else:
                self.db_manager = DatabaseManager(db_path)

            # Initialize other components
            self.keyword_searcher = KeywordSearcher(self.db_manager)
            self.semantic_searcher = SemanticSearcher(self.db_manager)
            self.mistral_client = get_mistral_client()
            self.embedding_manager = get_embedding_manager(self.db_manager)

            # Initialize query service
            self.query_service = RAGQueryService(
                db_path=db_path,
                mistral_client=self.mistral_client,
                db_manager=self.db_manager,
            )

        except Exception as e:
            st.error(f"Failed to initialize local components: {e}")
            logger.error(f"Local component initialization error: {e}")

    def render_sidebar(self):
        """Render the sidebar with upload and configuration options."""
        with st.sidebar:
            st.markdown("## 📚 Simple RAG System")

            # Mode indicator
            if self.use_backend:
                st.info(f"🌐 Backend Mode: {self.backend_url}")
            else:
                st.info("🏠 Local Mode: Using SQLite")

            st.markdown("---")

            # File upload section
            st.markdown("### 📄 Upload PDFs")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF files to build your knowledge base",
            )

            if uploaded_files and st.button("📥 Ingest Documents", type="primary"):
                self._handle_file_upload(uploaded_files)

            st.markdown("---")

            # Retrieval parameters
            st.markdown("### ⚙️ Retrieval Settings")

            top_k = st.slider(
                "Results to return",
                min_value=1,
                max_value=20,
                value=8,
                help="Number of final results to return",
            )

            rerank_k = st.slider(
                "Reranking candidates",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of candidates to consider for reranking",
            )

            threshold = st.slider(
                "Evidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.18,
                step=0.01,
                help="Minimum semantic similarity threshold",
            )

            alpha = st.slider(
                "Semantic vs Keyword weight",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                help="Alpha: 1.0 = pure semantic, 0.0 = pure keyword",
            )

            use_mmr = st.checkbox(
                "Use MMR diversification",
                value=True,
                help="Apply Maximal Marginal Relevance for result diversification",
            )

            # Store parameters in session state
            st.session_state.retrieval_params = {
                "top_k": top_k,
                "rerank_k": rerank_k,
                "threshold": threshold,
                "alpha": alpha,
                "use_mmr": use_mmr,
            }

            st.markdown("---")

            # System status
            self._render_system_status()

    def _render_system_status(self):
        """Render system status information."""
        st.markdown("### 📊 System Status")

        try:
            if self.use_backend:
                # Check backend health
                async def check_backend_health():
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"{self.backend_url}/health", timeout=5
                        )
                        if response.status_code == HTTP_OK:
                            return response.json()
                        return None

                health_data = asyncio.run(check_backend_health())
                if health_data:
                    st.success("✅ Backend connected")
                    st.metric("Documents", health_data.get("total_documents", 0))
                    st.metric("Chunks", health_data.get("total_chunks", 0))
                else:
                    st.error("❌ Backend unavailable")
            # Check local database
            elif hasattr(self, "db_manager") and self.db_manager is not None:
                with sqlite3.connect(self.db_manager.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM documents")
                    docs_count = cursor.fetchone()[0]
                    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                    chunks_count = cursor.fetchone()[0]

                st.success("✅ Database connected")
                st.metric("Documents", docs_count)
                st.metric("Chunks", chunks_count)
            else:
                st.warning("⚠️ Database not initialized")

        except Exception as e:
            st.error(f"❌ Status check failed: {e}")

    def _handle_file_upload(self, uploaded_files: list[Any]):
        """Handle PDF file upload and ingestion."""
        with st.spinner("Processing documents..."):
            try:
                if self.use_backend:
                    # Use backend API
                    async def upload_files():
                        files = []
                        for uploaded_file in uploaded_files:
                            files.append(
                                (
                                    "files",
                                    (
                                        uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        "application/pdf",
                                    ),
                                )
                            )

                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                f"{self.backend_url}/ingest", files=files, timeout=120
                            )
                            if response.status_code == HTTP_OK:
                                return response.json()
                            return {"error": response.text}

                    result = asyncio.run(upload_files())
                    if "error" not in result:
                        st.success(
                            f"✅ Successfully processed {len(result['documents'])} documents with {result['chunks_indexed']} chunks!"
                        )
                    else:
                        st.error(f"❌ Upload failed: {result['error']}")

                else:
                    # Use local processing
                    if self.db_manager is None or self.embedding_manager is None:
                        st.error("Local components not initialized")
                        return

                    total_chunks = 0
                    processed_docs = 0

                    for uploaded_file in uploaded_files:
                        try:
                            # Extract text
                            file_content = uploaded_file.getvalue()
                            pages_text, total_pages = extract_pdf_text(file_content)

                            if not pages_text:
                                st.warning(f"No text found in {uploaded_file.name}")
                                continue

                            # Add document
                            doc_id = self.db_manager.add_document(
                                uploaded_file.name, total_pages
                            )

                            # Chunk text
                            chunks = chunk_text(pages_text)

                            # Process chunks
                            chunk_data = []
                            for chunk in chunks:
                                chunk_id = self.db_manager.add_chunk(
                                    doc_id=doc_id,
                                    page=chunk["page"],
                                    position=chunk["position"],
                                    text=chunk["text"],
                                )

                                # Index for keyword search
                                assert self.keyword_searcher is not None
                                self.keyword_searcher.index_chunk(
                                    chunk_id, chunk["text"]
                                )

                                # Prepare for embedding
                                chunk_data.append(
                                    {"chunk_id": chunk_id, "text": chunk["text"]}
                                )

                            # Generate embeddings
                            if chunk_data:
                                self.embedding_manager.embed_and_store_chunks(
                                    chunk_data
                                )

                            total_chunks += len(chunks)
                            processed_docs += 1

                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                            continue

                    if processed_docs > 0:
                        # Rebuild df values after processing all documents
                        try:
                            st.info("Rebuilding document frequency indices...")
                            self.db_manager.rebuild_df()
                        except Exception as e:
                            st.warning(f"⚠️ Failed to rebuild df indices: {e}")
                            # Continue anyway - this is an optimization, not critical

                        st.success(
                            f"✅ Successfully processed {processed_docs} documents with {total_chunks} chunks!"
                        )
                        # Clear cache to update status
                        if (
                            hasattr(self, "semantic_searcher")
                            and self.semantic_searcher is not None
                        ):
                            self.semantic_searcher.clear_cache()
                    else:
                        st.error("❌ No documents were successfully processed")

            except Exception as e:
                st.error(f"❌ Error during upload: {e}")
                logger.error(f"Upload error: {e}")

    def _clear_database(self):
        """Clear all data from the database."""
        try:
            if self.db_manager is not None:
                self.db_manager.clear_all_data()
                # Clear chat history as well since context is now invalid
                st.session_state.messages = []
                # Clear any caches
                if self.semantic_searcher is not None:
                    self.semantic_searcher.clear_cache()
        except Exception as e:
            st.error(f"❌ Error clearing database: {e}")
            logger.error(f"Database clear error: {e}")

    def _clear_database_backend(self):
        """Clear all data from the backend database via API."""
        try:

            async def clear_backend_data():
                async with httpx.AsyncClient() as client:
                    response = await client.delete(
                        f"{self.backend_url}/collections", timeout=30
                    )
                    if response.status_code == HTTP_OK:
                        return response.json()
                    if response.status_code == HTTP_NOT_FOUND:
                        return {"error": "No collections found to delete"}
                    return {"error": response.text}

            result = asyncio.run(clear_backend_data())
            if "error" not in result:
                # Clear chat history as well since context is now invalid
                st.session_state.messages = []
            else:
                st.error(f"❌ Backend clear failed: {result['error']}")
        except Exception as e:
            st.error(f"❌ Error clearing backend database: {e}")
            logger.error(f"Backend database clear error: {e}")

    def _query_backend(
        self, query: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Query the backend API."""
        try:

            async def query_async():
                payload = {"query": query, **params}

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.backend_url}/query", json=payload, timeout=60
                    )

                    if response.status_code == HTTP_OK:
                        return response.json()
                    return {"error": response.text}

            result = asyncio.run(query_async())
            if "error" in result:
                st.error(f"Query failed: {result['error']}")
                return None
            return result

        except Exception as e:
            st.error(f"Error querying backend: {e}")
            return None

    def _query_local(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Query using local components via the unified query service."""
        if self.query_service is None:
            return {"error": "Query service not initialized"}

        try:
            # Prepare conversation history from Streamlit session state
            conversation_history = []
            if "conversation_history" in st.session_state:
                # Convert to ConversationMessage objects (last 6 messages = 3 exchanges)
                history = st.session_state.conversation_history[-6:]
                conversation_history = [
                    ConversationMessage(role=msg["role"], content=msg["content"])
                    for msg in history
                ]

            # Use the unified query service
            result = self.query_service.query(
                query=query,
                conversation_history=conversation_history,
                top_k=params.get("top_k", 8),
                rerank_k=params.get("rerank_k", 20),
                threshold=params.get("threshold", 0.18),
                alpha=params.get("alpha", 0.65),
                lambda_param=params.get("lambda_param", 0.7),
                use_mmr=params.get("use_mmr", True),
            )

            if not result.success:
                st.error(f"Query service error: {result.error}")
                return None

            # Convert result to expected format
            return {
                "answer": result.answer,
                "citations": [
                    {
                        "chunk_id": c.chunk_id,
                        "filename": c.filename,
                        "page": c.page,
                        "snippet": c.snippet,
                    }
                    for c in result.citations
                ],
                "insufficient_evidence": result.insufficient_evidence,
                "intent": result.intent,
            }

        except Exception as e:
            st.error(f"Local query error: {e}")
            logger.error(f"Local query error: {e}")
            return None

    def render_chat_interface(self):
        """Render the main chat interface."""
        st.markdown(
            '<h1 class="main-header">Simple RAG System</h1>', unsafe_allow_html=True
        )

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Display citations if present
                if message.get("citations"):
                    with st.expander(f"📚 Citations ({len(message['citations'])})"):
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(
                                f"""
                            <div class="citation-box">
                                <strong>{i}. {citation["filename"]} (p. {citation["page"]})</strong><br>
                                {citation["snippet"]}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with (
                st.chat_message("assistant"),
                st.spinner("Searching and generating response..."),
            ):
                # Get retrieval parameters
                params = st.session_state.get("retrieval_params", {})

                # Query the system
                if self.use_backend:
                    result = self._query_backend(prompt, params)
                else:
                    result = self._query_local(prompt, params)

                if result:
                    answer = result["answer"]
                    citations = result.get("citations", [])

                    # Display answer without special formatting
                    st.markdown(answer)

                    # Add to message history
                    message_data = {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                    st.session_state.messages.append(message_data)

                    # Display citations
                    if citations:
                        with st.expander(f"📚 Citations ({len(citations)})"):
                            for i, citation in enumerate(citations, 1):
                                st.markdown(
                                    f"""
                                <div class="citation-box">
                                    <strong>{i}. {citation["filename"]} (p. {citation["page"]})</strong><br>
                                    {citation["snippet"]}
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                else:
                    error_msg = (
                        "Sorry, I encountered an error processing your question."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    def run(self):
        """Main application runner."""
        # Render sidebar
        self.render_sidebar()

        # Render main chat interface
        self.render_chat_interface()

        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Clear database button
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🗃️ Database Management")
        
        # Show confirmation checkbox first
        confirm_delete = st.sidebar.checkbox(
            "I understand this will delete all data", key="confirm_delete"
        )
        
        # Only show the button if confirmed, and handle the click
        if confirm_delete and st.sidebar.button(
            "🗑️ Clear All Data",
            type="secondary",
            help="Delete all documents and chunks from the database",
        ):
            if self.use_backend:
                self._clear_database_backend()
            else:
                self._clear_database()
            st.sidebar.success("✅ Database cleared!")
            st.rerun()


def main():
    """Main entry point for the Streamlit app."""
    print(config)
    app = StreamlitRAG()
    app.run()


if __name__ == "__main__":
    main()
