"""FastAPI endpoint for document ingestion."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import logging
from datetime import datetime

from app.core.models import IngestResponse, DocumentInfo, ErrorResponse
from app.retriever.index import DatabaseManager
from app.ingest.pdf_extractor import extract_pdf_text
from app.ingest.chunker import chunk_text
from app.retriever.keyword import KeywordSearcher
from app.llm.mistral_client import get_embedding_manager
from app.core.config import config

logger = logging.getLogger(__name__)

router = APIRouter()


def get_db_manager():
    """Dependency to get database manager."""
    return DatabaseManager(config.DB_PATH)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Ingest PDF documents into the RAG system.
    
    This endpoint:
    1. Extracts text from uploaded PDFs
    2. Chunks the text with overlap
    3. Generates embeddings using Mistral API
    4. Indexes both TF-IDF and semantic embeddings
    5. Stores everything in SQLite database
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate Mistral API configuration
    if not config.validate_mistral_config():
        raise HTTPException(
            status_code=500, 
            detail="Mistral API key not configured. Please set MISTRAL_API_KEY environment variable."
        )
    
    ingested_documents = []
    total_chunks = 0
    
    try:
        # Initialize searchers and embedding manager
        keyword_searcher = KeywordSearcher(db_manager)
        embedding_manager = get_embedding_manager(db_manager)
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                logger.warning(f"Skipping non-PDF file: {file.filename}")
                continue
            
            logger.info(f"Processing file: {file.filename}")
            
            # Read file content
            file_content = await file.read()
            
            # Extract text from PDF
            try:
                pages_text, total_pages = extract_pdf_text(file_content)
            except Exception as e:
                logger.error(f"Failed to extract text from {file.filename}: {e}")
                continue
            
            if not pages_text:
                logger.warning(f"No text extracted from {file.filename}")
                continue
            
            # Add document to database
            doc_id = db_manager.add_document(file.filename, total_pages)
            
            # Chunk the text
            chunks = chunk_text(pages_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            
            if not chunks:
                logger.warning(f"No chunks created from {file.filename}")
                continue
            
            # Add chunks to database and collect for embedding
            chunk_data = []
            
            for chunk in chunks:
                chunk_id = db_manager.add_chunk(
                    doc_id=doc_id,
                    page=chunk["page"],
                    position=chunk["position"],
                    text=chunk["text"]
                )
                
                # Index for keyword search
                keyword_searcher.index_chunk(chunk_id, chunk["text"])
                
                # Prepare for embedding
                chunk_data.append({
                    "chunk_id": chunk_id,
                    "text": chunk["text"]
                })
            
            # Generate and store embeddings
            if chunk_data:
                try:
                    embedding_manager.embed_and_store_chunks(chunk_data)
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for {file.filename}: {e}")
                    # Continue processing other files even if embeddings fail
            
            # Record successful ingestion
            ingested_documents.append(DocumentInfo(
                id=doc_id,
                filename=file.filename,
                pages=total_pages,
                uploaded_at=datetime.now()
            ))
            
            total_chunks += len(chunks)
            logger.info(f"Successfully ingested {file.filename}: {len(chunks)} chunks")
        
        if not ingested_documents:
            raise HTTPException(
                status_code=400, 
                detail="No valid PDF documents could be processed"
            )
        
        return IngestResponse(
            documents=ingested_documents,
            chunks_indexed=total_chunks,
            success=True,
            message=f"Successfully ingested {len(ingested_documents)} documents with {total_chunks} chunks"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/ingest/status")
async def get_ingestion_status(db_manager: DatabaseManager = Depends(get_db_manager)):
    """Get status of the ingestion system."""
    try:
        # Get database statistics
        with db_manager.db_path as path:
            import sqlite3
            with sqlite3.connect(path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                total_docs = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                total_embeddings = cursor.fetchone()[0]
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "mistral_configured": config.validate_mistral_config()
        }
    
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ingestion status")