"""System status and health check components."""

import asyncio
import sqlite3
import streamlit as st
import httpx
from typing import Optional, Dict, Any


class SystemStatus:
    """Manages system status checks and displays."""
    
    def __init__(self, backend_url: Optional[str] = None, db_manager=None):
        self.backend_url = backend_url
        self.use_backend = bool(backend_url)
        self.db_manager = db_manager

    async def check_backend_health(self) -> Optional[Dict[str, Any]]:
        """Check backend API health status."""
        if not self.backend_url:
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/health", timeout=5)
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return None

    def get_local_stats(self) -> Dict[str, int]:
        """Get local database statistics."""
        if not self.db_manager:
            return {"documents": 0, "chunks": 0}
            
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                docs_count = cursor.fetchone()[0]
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                chunks_count = cursor.fetchone()[0]
                
            return {"documents": docs_count, "chunks": chunks_count}
        except Exception:
            return {"documents": 0, "chunks": 0}

    def render_status_metrics(self):
        """Render system status with metrics."""
        st.markdown("### 📊 System Status")
        
        try:
            if self.use_backend:
                # Check backend status
                health_data = asyncio.run(self.check_backend_health())
                if health_data:
                    st.success("✅ Backend connected")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Documents", health_data.get("total_documents", 0))
                    with col2:
                        st.metric("Chunks", health_data.get("total_chunks", 0))
                else:
                    st.error("❌ Backend unavailable")
                    st.metric("Status", "Offline")
                    
            elif self.db_manager:
                # Check local database
                stats = self.get_local_stats()
                st.success("✅ Database connected")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", stats["documents"])
                with col2:
                    st.metric("Chunks", stats["chunks"])
            else:
                st.warning("⚠️ Database not initialized")
                
        except Exception as e:
            st.error(f"❌ Status check failed: {str(e)}")

    def render_connection_status(self):
        """Render just the connection status indicator."""
        if self.use_backend:
            health_data = asyncio.run(self.check_backend_health())
            if health_data:
                st.success(f"🌐 Backend: Connected")
                # Show additional backend info
                if "status" in health_data:
                    status = health_data["status"]
                    if status == "healthy":
                        st.success("✅ Backend is healthy")
                    elif status == "degraded":
                        st.warning("⚠️ Backend is degraded")
                    else:
                        st.info(f"ℹ️ Backend status: {status}")
            else:
                st.error(f"🌐 Backend: Offline")
                st.error("❌ Cannot connect to backend - check if it's running")
                with st.expander("Backend Connection Help"):
                    st.markdown("""
                    **Troubleshooting Backend Connection:**
                    1. Make sure the FastAPI backend is running:
                       ```bash
                       uv run uvicorn src.app.main:app --reload --port 8000
                       ```
                    2. Check that BACKEND_URL is set correctly:
                       ```bash
                       export BACKEND_URL=http://localhost:8000
                       ```
                    3. Verify the backend is accessible at the URL
                    """)
        elif self.db_manager:
            st.info("🏠 Local: SQLite")
        else:
            st.warning("⚠️ Not initialized")


def render_mode_indicator(backend_url: Optional[str]):
    """Render the system mode indicator."""
    if backend_url:
        st.info(f"🌐 Backend Mode: {backend_url}")
    else:
        st.info("🏠 Local Mode: Using SQLite")


def show_database_stats(db_manager=None, backend_url: Optional[str] = None):
    """Show database statistics in a compact format."""
    if backend_url:
        # Backend mode
        try:
            async def get_backend_stats():
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{backend_url}/health", timeout=3)
                    return response.json() if response.status_code == 200 else None
            
            stats = asyncio.run(get_backend_stats())
            if stats:
                docs = stats.get("total_documents", 0)
                chunks = stats.get("total_chunks", 0)
                st.write(f"📊 {docs} documents, {chunks} chunks")
                if docs == 0:
                    st.info("💡 Upload some PDFs to get started!")
            else:
                st.write("📊 Backend unavailable")
                st.error("❌ Cannot connect to backend")
        except Exception as e:
            st.write("📊 Backend error")
            st.error(f"❌ Backend error: {str(e)}")
    
    elif db_manager:
        # Local mode
        try:
            with sqlite3.connect(db_manager.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                docs = cursor.fetchone()[0]
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                chunks = cursor.fetchone()[0]
            st.write(f"📊 {docs} documents, {chunks} chunks")
            if docs == 0:
                st.info("💡 Upload some PDFs to get started!")
        except Exception as e:
            st.write("📊 Database error")
            st.error(f"❌ Database error: {str(e)}")