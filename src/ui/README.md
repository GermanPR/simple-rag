# UI Module Structure

This directory contains the modular Streamlit UI components for the RAG system.

## Structure

```
ui/
├── streamlit_app.py          # Main entry point (~140 lines)
├── components/               # Reusable UI components
│   ├── chat.py              # Chat interface with improved message handling
│   ├── sidebar.py           # Clean sidebar with upload & settings  
│   ├── status.py            # System status and health checks
│   └── progress.py          # Progress indicators for operations
├── handlers/                # Business logic handlers
│   ├── upload_handler.py    # Non-blocking file upload processing
│   └── query_handler.py     # Smart query processing with progress
├── utils/                   # Utilities
│   ├── session_state.py     # Session state management
│   └── ui_helpers.py        # Common UI helper functions
└── styles/
    └── css.py               # Custom CSS styles
```

## Key Improvements

### Non-Blocking Operations
- **Document Upload**: Uses `st.status()` with progress tracking instead of blocking `st.spinner()`
- **Query Processing**: Shows progressive steps without grey overlay
- **Better UX**: Users can see what's happening in real-time

### Smart Chat Management
- **Message Pairs**: Only commits complete user-assistant exchanges to history
- **Error Handling**: Graceful error display without breaking chat flow
- **Progressive Loading**: Shows "Processing query"

### Clean Interface
- **Hidden Advanced Settings**: Retrieval parameters moved to collapsible section
- **Simplified Sidebar**: Focus on core functionality
- **Professional Appearance**: Clean, minimal design

## Usage

Run the application:
```bash
# Local mode
uv run streamlit run src/ui/streamlit_app.py

# Backend mode  
BACKEND_URL=http://localhost:8000 uv run streamlit run src/ui/streamlit_app.py
```

## Benefits

1. **Maintainability**: Each component has a single responsibility
2. **Reusability**: Components can be easily reused or swapped
3. **Testability**: Individual components can be tested in isolation
4. **Better UX**: No more blocking UI operations
5. **Professional**: Clean, responsive interface
