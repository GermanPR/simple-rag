"""Session state management utilities for the RAG UI."""

import streamlit as st
from typing import Any


def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    if "retrieval_params" not in st.session_state:
        st.session_state.retrieval_params = {
            "top_k": 8,
            "rerank_k": 20,
            "threshold": 0.18,
            "alpha": 0.65,
            "use_mmr": True,
        }
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def get_retrieval_params() -> dict[str, Any]:
    """Get current retrieval parameters."""
    return st.session_state.get("retrieval_params", {
        "top_k": 8,
        "rerank_k": 20,
        "threshold": 0.18,
        "alpha": 0.65,
        "use_mmr": True,
    })


def update_retrieval_params(params: dict[str, Any]):
    """Update retrieval parameters in session state."""
    st.session_state.retrieval_params.update(params)


def add_message_pair(user_message: str, assistant_message: str, citations: list = None):
    """Add a complete user-assistant message pair to chat history."""
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": user_message
    })
    
    # Add assistant message with citations
    assistant_msg = {
        "role": "assistant",
        "content": assistant_message,
    }
    if citations:
        assistant_msg["citations"] = citations
    
    st.session_state.messages.append(assistant_msg)
    
    # Update conversation history for context
    st.session_state.conversation_history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ])


def clear_chat_history():
    """Clear chat history and conversation context."""
    st.session_state.messages = []
    st.session_state.conversation_history = []


def get_conversation_context() -> list:
    """Get recent conversation history for context (last 6 messages = 3 exchanges)."""
    return st.session_state.conversation_history[-6:]