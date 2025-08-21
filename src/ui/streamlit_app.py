"""Streamlit UI for the RAG system - Modular version."""

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

# Add app directory to path
app_root = Path(__file__).parent.parent
sys.path.insert(0, str(app_root))

# Core imports  # noqa: E402
from app.core.config import config  # noqa: E402
from app.core.logging_config import get_logger, setup_logging  # noqa: E402
from app.llm.mistral_client import get_embedding_manager, get_mistral_client  # noqa: E402
from app.retriever.index import DatabaseManager  # noqa: E402
from app.retriever.keyword import KeywordSearcher  # noqa: E402
from app.retriever.semantic import SemanticSearcher  # noqa: E402
from app.services.query_service import RAGQueryService  # noqa: E402

# UI Component imports  # noqa: E402
from ui.components.chat import create_chat_interface  # noqa: E402
from ui.components.sidebar import create_sidebar  # noqa: E402
from ui.handlers.query_handler import QueryHandler  # noqa: E402
from ui.styles.css import apply_custom_styles, get_page_config  # noqa: E402
from ui.utils.session_state import get_retrieval_params, init_session_state  # noqa: E402

# Configure logging
setup_logging()
logger = get_logger(__name__.split(".")[-1])


class StreamlitRAG:
    """Main RAG application with modular architecture."""

    def __init__(self):
        # Configuration with debug logging - check both env vars and Streamlit secrets
        self.backend_url = os.getenv("BACKEND_URL") or st.secrets.get("BACKEND_URL", None)
        self.use_backend = bool(self.backend_url)
        
        # Debug logging for backend detection
        env_backend_url = os.getenv("BACKEND_URL")
        secrets_backend_url = st.secrets.get("BACKEND_URL", None)
        logger.info(f"BACKEND_URL from environment: '{env_backend_url}'")
        logger.info(f"BACKEND_URL from secrets: '{secrets_backend_url}'")
        logger.info(f"Final BACKEND_URL: '{self.backend_url}'")
        logger.info(f"Using backend mode: {self.use_backend}")
        
        # Validate backend URL if provided
        if self.backend_url and self.backend_url.strip():
            if not (self.backend_url.startswith('http://') or self.backend_url.startswith('https://')):
                logger.warning(f"Backend URL doesn't start with http:// or https://: {self.backend_url}")
                self.use_backend = False
            else:
                logger.info(f"Backend URL appears valid: {self.backend_url}")
                # Test backend connectivity
                self._test_backend_connectivity()
        elif self.backend_url == "":
            logger.info("Backend URL is empty string - using local mode")
            self.use_backend = False

        # Initialize session state
        init_session_state()

        # Component instances
        self.db_manager = None
        self.query_service = None
        self.sidebar = None
        self.chat_interface = None

        # Initialize components
        if not self.use_backend:
            self._init_local_components()

        self._init_ui_components()

    def _test_backend_connectivity(self):
        """Test backend connectivity and update use_backend flag accordingly."""
        try:
            logger.info(f"Testing connectivity to backend: {self.backend_url}")
            response = httpx.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Backend connectivity test successful")
                backend_info = response.json()
                logger.info(f"Backend status: {backend_info.get('status', 'unknown')}")
                logger.info(f"Documents: {backend_info.get('total_documents', 0)}, Chunks: {backend_info.get('total_chunks', 0)}")
            else:
                logger.warning(f"Backend health check returned status {response.status_code}")
                logger.info("Falling back to local mode due to backend connectivity issues")
                self.use_backend = False
        except Exception as e:
            logger.error(f"Backend connectivity test failed: {e}")
            logger.info("Falling back to local mode due to backend connectivity failure")
            self.use_backend = False

    def _init_local_components(self):
        """Initialize local RAG components."""
        try:
            # Database setup
            db_path = st.secrets.get("DB_PATH", config.DB_PATH)

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

            # Query service
            self.query_service = RAGQueryService(
                db_path=db_path,
                mistral_client=self.mistral_client,
                db_manager=self.db_manager,
            )

        except Exception as e:
            st.error(f"Failed to initialize local components: {e}")
            logger.error(f"Local component initialization error: {e}")

    def _init_ui_components(self):
        """Initialize UI components."""
        # Create query handler
        query_handler = QueryHandler(
            backend_url=self.backend_url,
            query_service=self.query_service
        )

        # Create sidebar
        sidebar_kwargs = {}
        if not self.use_backend:
            sidebar_kwargs.update({
                "db_manager": self.db_manager,
                "embedding_manager": self.embedding_manager,
                "keyword_searcher": getattr(self, "keyword_searcher", None),
                "semantic_searcher": getattr(self, "semantic_searcher", None),
            })

        self.sidebar = create_sidebar(
            backend_url=self.backend_url,
            **sidebar_kwargs
        )

        # Create chat interface
        self.chat_interface = create_chat_interface(query_handler)

    def run(self):
        """Run the main application."""
        # Apply styling
        apply_custom_styles()

        # Render sidebar
        self.sidebar.render_complete_sidebar()

        # Render main chat interface
        st.markdown("# 💬 Chat with your documents")

        retrieval_params = get_retrieval_params()
        self.chat_interface.render_main_interface(retrieval_params)



def main():
    """Main entry point."""
    # Page configuration
    page_config = get_page_config()
    st.set_page_config(**page_config)

    # Create and run app
    app = StreamlitRAG()
    app.run()


if __name__ == "__main__":
    main()
