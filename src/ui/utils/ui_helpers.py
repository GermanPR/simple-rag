"""UI helper utilities."""

import streamlit as st
from typing import Any, Dict


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def show_info_box(message: str, icon: str = "ℹ️"):
    """Display an info box with custom styling."""
    st.markdown(
        f"""
        <div style="
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        ">
            {icon} {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_success_box(message: str, icon: str = "✅"):
    """Display a success box with custom styling."""
    st.markdown(
        f"""
        <div style="
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            color: #2e7d2e;
        ">
            {icon} {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_error_box(message: str, icon: str = "❌"):
    """Display an error box with custom styling.""" 
    st.markdown(
        f"""
        <div style="
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            color: #c62828;
        ">
            {icon} {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def create_two_column_layout():
    """Create a standard two-column layout."""
    return st.columns([3, 1])


def create_centered_container():
    """Create a centered container with padding."""
    col1, col2, col3 = st.columns([1, 2, 1])
    return col2