"""Non-blocking upload handler for PDF files."""

import asyncio
import httpx
import streamlit as st
from typing import Any, List, Optional

from ..components.progress import upload_progress
from ..utils.session_state import get_retrieval_params


class UploadHandler:
    """Handles PDF file upload with progress tracking."""
    
    def __init__(self, backend_url: Optional[str] = None, 
                 db_manager=None, embedding_manager=None, 
                 keyword_searcher=None):
        self.backend_url = backend_url
        self.use_backend = bool(backend_url)
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager  
        self.keyword_searcher = keyword_searcher

    async def upload_to_backend(self, uploaded_files: List[Any]) -> dict:
        """Upload files to backend API."""
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
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}

    def process_local_file(self, uploaded_file, update_progress, complete_file):
        """Process a single file locally with progress updates."""
        from app.ingest.pdf_extractor import extract_pdf_text
        from app.ingest.chunker import chunk_text
        
        try:
            # Step 1: Extract text
            update_progress(uploaded_file.name, "Extracting text...")
            file_content = uploaded_file.getvalue()
            pages_text, total_pages = extract_pdf_text(file_content)

            if not pages_text:
                raise ValueError("No text found in PDF")

            # Step 2: Add document to database
            update_progress(uploaded_file.name, "Creating document record...")
            doc_id = self.db_manager.add_document(uploaded_file.name, total_pages)

            # Step 3: Chunk text
            update_progress(uploaded_file.name, "Chunking text...")
            chunks = chunk_text(pages_text)

            # Step 4: Process chunks
            update_progress(uploaded_file.name, "Processing chunks...")
            chunk_data = []
            for chunk in chunks:
                chunk_id = self.db_manager.add_chunk(
                    doc_id=doc_id,
                    page=chunk["page"],
                    position=chunk["position"],
                    text=chunk["text"],
                )

                # Index for keyword search
                if self.keyword_searcher:
                    self.keyword_searcher.index_chunk(chunk_id, chunk["text"])

                # Prepare for embedding
                chunk_data.append(
                    {"chunk_id": chunk_id, "text": chunk["text"]}
                )

            # Step 5: Generate embeddings
            if chunk_data and self.embedding_manager:
                update_progress(uploaded_file.name, "Generating embeddings...")
                self.embedding_manager.embed_and_store_chunks(chunk_data)

            complete_file()
            return len(chunks)

        except Exception as e:
            raise Exception(f"Error processing {uploaded_file.name}: {str(e)}")

    def handle_upload(self, uploaded_files: List[Any]) -> bool:
        """Handle file upload with progress tracking."""
        if not uploaded_files:
            return False

        try:
            if self.use_backend:
                # Backend upload
                with upload_progress(len(uploaded_files)) as (update_file, complete_file, set_error):
                    try:
                        for file in uploaded_files:
                            update_file(file.name, "Uploading...")
                        
                        # Upload all files at once
                        result = asyncio.run(self.upload_to_backend(uploaded_files))
                        
                        if "error" not in result:
                            for file in uploaded_files:
                                update_file(file.name, "Processed successfully")
                                complete_file()
                            
                            st.success(
                                f"✅ Successfully processed {len(result['documents'])} documents "
                                f"with {result['chunks_indexed']} chunks!"
                            )
                            return True
                        else:
                            set_error(result["error"])
                            return False
                            
                    except Exception as e:
                        set_error(str(e))
                        return False

            else:
                # Local processing
                if not self._validate_local_components():
                    st.error("Local components not properly initialized")
                    return False

                total_chunks = 0
                processed_docs = 0

                with upload_progress(len(uploaded_files)) as (update_file, complete_file, set_error):
                    try:
                        for uploaded_file in uploaded_files:
                            chunks_count = self.process_local_file(
                                uploaded_file, update_file, complete_file
                            )
                            total_chunks += chunks_count
                            processed_docs += 1

                        # Rebuild document frequency indices
                        if processed_docs > 0:
                            st.info("Rebuilding search indices...")
                            self.db_manager.rebuild_df()

                            # Clear semantic searcher cache
                            if hasattr(self, 'semantic_searcher') and self.semantic_searcher:
                                self.semantic_searcher.clear_cache()

                            st.success(
                                f"✅ Successfully processed {processed_docs} documents "
                                f"with {total_chunks} chunks!"
                            )
                            return True
                        else:
                            set_error("No documents were successfully processed")
                            return False

                    except Exception as e:
                        set_error(str(e))
                        return False

        except Exception as e:
            st.error(f"❌ Upload error: {str(e)}")
            return False

    def _validate_local_components(self) -> bool:
        """Validate that all required local components are available."""
        if not self.db_manager:
            st.error("Database manager not initialized")
            return False
        if not self.embedding_manager:
            st.error("Embedding manager not initialized")
            return False
        return True