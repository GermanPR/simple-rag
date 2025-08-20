"""FastAPI endpoint for collection management."""

import logging
import os
from typing import Any

import aiosqlite
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status

from app.core.config import config
from app.core.models import MessageResponse
from app.retriever.index import AsyncDatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_async_db_manager():
    """Dependency to get async database manager."""
    db_manager = AsyncDatabaseManager(config.DB_PATH)
    await db_manager.init_database()
    return db_manager


@router.delete("/collections", response_model=MessageResponse)
async def delete_all_collections(
    db_manager: AsyncDatabaseManager = Depends(get_async_db_manager),
) -> MessageResponse:
    """
    Delete all ingested collections (documents, chunks, embeddings, indices).

    This endpoint will:
    1. Remove all documents and their associated data
    2. Clear all text chunks and embeddings
    3. Reset TF-IDF indices and keyword search data
    4. Preserve database schema for future ingestion

    **WARNING**: This operation is irreversible and will permanently delete
    all ingested data. Use with caution.

    Returns:
        MessageResponse: Confirmation with details of deleted data

    Raises:
        HTTPException: 500 if deletion fails
        HTTPException: 404 if no data exists to delete
    """
    try:
        # First check if there's any data to delete
        async with aiosqlite.connect(db_manager.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM documents")
            total_docs_result = await cursor.fetchone()
            total_docs = total_docs_result[0] if total_docs_result else 0

            cursor = await conn.execute("SELECT COUNT(*) FROM chunks")
            total_chunks_result = await cursor.fetchone()
            total_chunks = total_chunks_result[0] if total_chunks_result else 0

            cursor = await conn.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings_result = await cursor.fetchone()
            total_embeddings = (
                total_embeddings_result[0] if total_embeddings_result else 0
            )

        if total_docs == 0 and total_chunks == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No collections found to delete",
            )

        # Perform the deletion
        await db_manager.clear_all_data()

        logger.info(
            f"Successfully deleted all collections: {total_docs} documents, "
            f"{total_chunks} chunks, {total_embeddings} embeddings"
        )

        return MessageResponse(
            success=True,
            message=(
                f"Successfully deleted all collections. "
                f"Removed {total_docs} documents, {total_chunks} chunks, "
                f"and {total_embeddings} embeddings."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collections: {str(e)}",
        ) from e


@router.get("/collections/stats")
async def get_collection_stats(
    db_manager: AsyncDatabaseManager = Depends(get_async_db_manager),
) -> dict[str, Any]:
    """
    Get statistics about the current collections.

    Returns collection counts and storage information.

    Returns:
        dict: Statistics about documents, chunks, embeddings, and indices

    Raises:
        HTTPException: 500 if unable to retrieve stats
    """
    try:
        async with aiosqlite.connect(db_manager.db_path) as conn:
            # Get document stats
            cursor = await conn.execute(
                "SELECT COUNT(*) as total, COUNT(DISTINCT filename) as unique_files FROM documents"
            )
            doc_stats = await cursor.fetchone()

            # Get chunk stats
            cursor = await conn.execute("SELECT COUNT(*) FROM chunks")
            total_chunks_result = await cursor.fetchone()
            total_chunks = total_chunks_result[0] if total_chunks_result else 0

            # Get embedding stats
            cursor = await conn.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings_result = await cursor.fetchone()
            total_embeddings = (
                total_embeddings_result[0] if total_embeddings_result else 0
            )

            # Get TF-IDF index stats
            cursor = await conn.execute("SELECT COUNT(*) FROM tokens")
            total_tokens_result = await cursor.fetchone()
            total_tokens = total_tokens_result[0] if total_tokens_result else 0

            cursor = await conn.execute("SELECT COUNT(*) FROM df")
            vocabulary_size_result = await cursor.fetchone()
            vocabulary_size = vocabulary_size_result[0] if vocabulary_size_result else 0

            # Calculate database file size if possible
            try:
                db_size = os.path.getsize(db_manager.db_path)
                db_size_mb = db_size / (1024 * 1024)
            except OSError:
                db_size_mb = None

        return {
            "documents": {
                "total": doc_stats[0] if doc_stats else 0,
                "unique_files": doc_stats[1] if doc_stats else 0,
            },
            "chunks": {
                "total": total_chunks,
                "with_embeddings": total_embeddings,
            },
            "indices": {
                "token_entries": total_tokens,
                "vocabulary_size": vocabulary_size,
            },
            "storage": {
                "database_size_mb": round(db_size_mb, 2) if db_size_mb else None,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve collection statistics: {str(e)}",
        ) from e
