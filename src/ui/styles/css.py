"""Custom CSS styles for the RAG UI."""

import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(
        """
<style>
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
.processing-message {
    color: #ffa500;
    font-style: italic;
}
.chat-container {
    margin: 0.5rem 0;
}
.status-container {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 0.5rem;
    margin: 0.25rem 0;
}
</style>
""",
        unsafe_allow_html=True,
    )


def get_page_config():
    """Get the page configuration for Streamlit."""
    return {
        "page_title": "RAG System",
        "page_icon": "📚",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }