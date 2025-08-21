"""Improved chat interface with better message handling."""

import streamlit as st
from typing import Dict, Any, Optional

from ..utils.session_state import add_message_pair, clear_chat_history
from ..handlers.query_handler import QueryHandler


class ChatInterface:
    """Manages the chat interface and message flow."""
    
    def __init__(self, query_handler: QueryHandler):
        self.query_handler = query_handler

    def render_message(self, message: Dict[str, Any]):
        """Render a single chat message."""
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if present
            if message.get("citations"):
                self._render_citations(message["citations"])

    def _render_citations(self, citations: list):
        """Render citations in an expandable section."""
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

    def render_chat_history(self):
        """Render all chat messages."""
        for message in st.session_state.messages:
            self.render_message(message)

    def render_chat_input(self, retrieval_params: Dict[str, Any]):
        """Render chat input and handle new messages."""
        if prompt := st.chat_input("Ask a question about your documents..."):
            self._handle_new_message(prompt, retrieval_params)

    def _handle_new_message(self, user_input: str, params: Dict[str, Any]):
        """Handle a new user message with improved UX."""
        # Show user message immediately  
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Show assistant response with progress
        with st.chat_message("assistant"):
            # Create placeholder for response
            response_placeholder = st.empty()
            
            # Show processing indicator
            with response_placeholder.container():
                st.markdown("🤔 Thinking...")
            
            # Process query
            result = self.query_handler.handle_query(user_input, params)
            
            if result:
                answer = result["answer"]
                citations = result.get("citations", [])
                
                # Update placeholder with actual response
                with response_placeholder.container():
                    st.markdown(answer)
                    
                    # Show citations
                    if citations:
                        self._render_citations(citations)
                
                # Add complete conversation to history
                add_message_pair(user_input, answer, citations)
                
            else:
                # Handle error case
                error_msg = "Sorry, I encountered an error processing your question."
                with response_placeholder.container():
                    st.error(error_msg)
                
                # Add error to history
                add_message_pair(user_input, error_msg)
            
            # Force rerun to update message history display
            st.rerun()

    def render_chat_controls(self):
        """Render chat control buttons."""
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
                clear_chat_history()
                st.rerun()

    def render_main_interface(self, retrieval_params: Dict[str, Any]):
        """Render the complete chat interface."""
        # Chat history
        self.render_chat_history()
        
        # Chat input
        self.render_chat_input(retrieval_params)


def create_chat_interface(query_handler: QueryHandler) -> ChatInterface:
    """Factory function to create a chat interface."""
    return ChatInterface(query_handler)