"""Smart query handler with progress tracking."""

import asyncio
import httpx
import streamlit as st
from typing import Any, Dict, Optional

from app.logic.intent import ConversationMessage
from ..components.progress import query_progress
from ..utils.session_state import get_conversation_context


class QueryHandler:
    """Handles query processing with progress feedback."""
    
    def __init__(self, backend_url: Optional[str] = None, query_service=None):
        self.backend_url = backend_url
        self.use_backend = bool(backend_url)
        self.query_service = query_service

    async def query_backend_async(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query the backend API asynchronously."""
        payload = {"query": query, **params}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/query", json=payload, timeout=60
            )

            if response.status_code == 200:
                return response.json()
            return {"error": response.text}

    def query_local(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query using local components."""
        if not self.query_service:
            return {"error": "Query service not initialized"}

        try:
            # Get conversation history
            history = get_conversation_context()
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
                threshold=params.get("threshold", 0.4),
                alpha=params.get("alpha", 0.65),
                lambda_param=params.get("lambda_param", 0.7),
                use_mmr=params.get("use_mmr", True),
            )

            if not result.success:
                return {"error": f"Query service error: {result.error}"}

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
            return {"error": f"Local query error: {str(e)}"}

    def process_query_with_progress(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process query with progress tracking."""
        with query_progress() as (add_step, complete_step, set_complete, set_error):
            try:
                if self.use_backend:
                    # Backend processing with progress
                    add_step("Analyzing query", "🔍")
                    add_step("Searching documents", "🔎") 
                    add_step("Generating response", "🤖")
                    
                    # Process query
                    result = asyncio.run(self.query_backend_async(query, params))
                    
                    if "error" in result:
                        set_error(f"Backend error: {result['error']}")
                        return None
                    
                    # Complete all steps
                    complete_step(0, "✅")
                    complete_step(1, "✅") 
                    complete_step(2, "✅")
                    set_complete()
                    
                    return result

                else:
                    # Local processing with progress
                    add_step("Analyzing intent", "🔍")
                    complete_step(0, "✅")
                    
                    add_step("Searching documents", "🔎")
                    complete_step(1, "✅")
                    
                    add_step("Generating answer", "🤖")
                    
                    # Process query locally
                    result = self.query_local(query, params)
                    
                    if "error" in result:
                        set_error(result["error"])
                        return None
                    
                    complete_step(2, "✅")
                    set_complete()
                    
                    return result

            except Exception as e:
                set_error(f"Query processing failed: {str(e)}")
                return None

    def handle_query(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Main entry point for handling queries."""
        if not query.strip():
            st.error("Please enter a question")
            return None
            
        try:
            return self.process_query_with_progress(query, params)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return None