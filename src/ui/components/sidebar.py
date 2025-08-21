"""Clean sidebar component with upload and settings."""

import asyncio
import os
import httpx
import streamlit as st
from typing import Any, List, Optional

from ..components.status import SystemStatus, show_database_stats
from ..handlers.upload_handler import UploadHandler
from ..utils.session_state import get_retrieval_params, update_retrieval_params, clear_chat_history


class Sidebar:
    """Manages the sidebar interface."""
    
    def __init__(self, backend_url: Optional[str] = None, 
                 db_manager=None, embedding_manager=None, 
                 keyword_searcher=None, semantic_searcher=None):
        self.backend_url = backend_url
        self.use_backend = bool(backend_url)
        self.db_manager = db_manager
        self.semantic_searcher = semantic_searcher
        
        # Initialize components
        self.status_manager = SystemStatus(backend_url, db_manager)
        self.upload_handler = UploadHandler(
            backend_url, db_manager, embedding_manager, keyword_searcher
        )

    def render_header(self):
        """Render sidebar header with mode indicator."""
        st.markdown("## 📚 RAG System")
        
        # Show connection status with debug info
        if self.backend_url:
            st.info(f"🌐 Backend: {self.backend_url}")
            # Add debug expander for backend mode
            with st.expander("🔍 Debug Info", expanded=False):
                st.code(f"BACKEND_URL={os.getenv('BACKEND_URL', 'Not set')}")
                st.code(f"use_backend={self.use_backend}")
        else:
            st.info("🏠 Local SQLite")
            # Add debug expander for local mode  
            with st.expander("🔍 Debug Info", expanded=False):
                st.code(f"BACKEND_URL={os.getenv('BACKEND_URL', 'Not set')}")
                st.code(f"use_backend={self.use_backend}")
                from app.core.config import config
                st.code(f"DB_PATH={config.DB_PATH}")

    def render_file_upload(self):
        """Render file upload section."""
        st.markdown("### 📄 Upload PDFs")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF files to build your knowledge base",
        )

        if uploaded_files and st.button("📥 Ingest Documents", type="primary"):
            success = self.upload_handler.handle_upload(uploaded_files)
            if success:
                # Clear semantic searcher cache if available
                if self.semantic_searcher:
                    self.semantic_searcher.clear_cache()
                st.rerun()

    def render_advanced_settings(self):
        """Render advanced settings in collapsible section."""
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.markdown("**Retrieval Parameters**")
            
            current_params = get_retrieval_params()
            
            top_k = st.slider(
                "Results to return",
                min_value=1,
                max_value=20,
                value=current_params.get("top_k", 8),
                help="Number of final results to return",
            )

            rerank_k = st.slider(
                "Reranking candidates",
                min_value=5,
                max_value=50,
                value=current_params.get("rerank_k", 20),
                help="Number of candidates for reranking",
            )

            threshold = st.slider(
                "Evidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=current_params.get("threshold", 0.18),
                step=0.01,
                help="Minimum similarity threshold",
            )

            alpha = st.slider(
                "Semantic vs Keyword",
                min_value=0.0,
                max_value=1.0,
                value=current_params.get("alpha", 0.65),
                step=0.05,
                help="1.0 = pure semantic, 0.0 = pure keyword",
            )

            use_mmr = st.checkbox(
                "Use MMR diversification",
                value=current_params.get("use_mmr", True),
                help="Apply result diversification",
            )

            # Update parameters
            update_retrieval_params({
                "top_k": top_k,
                "rerank_k": rerank_k,
                "threshold": threshold,
                "alpha": alpha,
                "use_mmr": use_mmr,
            })

    def render_system_status(self):
        """Render compact system status."""
        st.markdown("### 📊 Status")
        show_database_stats(self.db_manager, self.backend_url)

    def render_chat_controls(self):
        """Render chat management controls."""
        st.markdown("### 💬 Chat")
        
        if st.button("🗑️ Clear Chat", help="Clear chat history"):
            clear_chat_history()
            st.rerun()

    def render_database_management(self):
        """Render database management section."""
        st.markdown("### 🗃️ Database")
        
        # Show confirmation checkbox
        confirm_delete = st.checkbox(
            "I understand this will delete all data",
            key="confirm_delete",
            help="Check this box to enable the clear data button"
        )
        
        # Only show button if confirmed
        if confirm_delete and st.button(
            "🗑️ Clear All Data",
            type="secondary", 
            help="Delete all documents and chunks",
        ):
            self._clear_database()
            st.rerun()

    def _clear_database(self):
        """Clear database with appropriate method."""
        try:
            if self.use_backend:
                self._clear_backend_database()
            else:
                self._clear_local_database()
            
            st.success("✅ Database cleared!")
            
        except Exception as e:
            st.error(f"❌ Error clearing database: {str(e)}")

    def _clear_backend_database(self):
        """Clear backend database via API."""
        async def clear_backend_data():
            async with httpx.AsyncClient() as client:
                response = await client.delete(f"{self.backend_url}/collections", timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return {"message": "No data to clear"}
                else:
                    raise Exception(response.text)

        result = asyncio.run(clear_backend_data())
        clear_chat_history()  # Clear chat since context is invalid

    def _clear_local_database(self):
        """Clear local database."""
        if self.db_manager:
            self.db_manager.clear_all_data()
            clear_chat_history()  # Clear chat since context is invalid
            
            # Clear caches
            if self.semantic_searcher:
                self.semantic_searcher.clear_cache()

    def render_complete_sidebar(self):
        """Render the complete sidebar interface."""
        with st.sidebar:
            self.render_header()
            st.markdown("---")
            
            self.render_file_upload()
            st.markdown("---")
            
            self.render_advanced_settings()
            st.markdown("---")
            
            self.render_system_status()
            st.markdown("---")
            
            self.render_chat_controls()
            st.markdown("---")
            
            self.render_database_management()


def create_sidebar(backend_url: Optional[str] = None, **kwargs) -> Sidebar:
    """Factory function to create a sidebar."""
    return Sidebar(backend_url=backend_url, **kwargs)