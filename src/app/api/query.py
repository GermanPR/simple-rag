"""FastAPI endpoint for query processing."""

import logging
import time

import aiosqlite
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from app.core.config import config
from app.core.models import QueryRequest
from app.core.models import QueryResponse
from app.llm.mistral_client import get_mistral_client
from app.logic.intent import ConversationMessage
from app.retriever.index import AsyncDatabaseManager
from app.services.query_service import AsyncRAGQueryService

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_async_db_manager():
    """Dependency to get async database manager."""
    db_manager = AsyncDatabaseManager(config.DB_PATH)
    await db_manager.init_database()
    return db_manager


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db_manager: AsyncDatabaseManager = Depends(get_async_db_manager),
):
    """
    Query the RAG system for an answer.

    This endpoint:
    1. Detects query intent and rewrites query
    2. Performs hybrid search (TF-IDF + semantic)
    3. Applies MMR diversification
    4. Generates answer using Mistral API
    5. Post-processes with citations and validation
    """
    start_time = time.time()

    # Validate Mistral API configuration
    if not config.validate_mistral_config():
        raise HTTPException(
            status_code=500,
            detail="Mistral API key not configured. Please set MISTRAL_API_KEY environment variable.",
        )

    try:
        # Initialize the async query service
        mistral_client = get_mistral_client()
        query_service = AsyncRAGQueryService(
            async_db_manager=db_manager,
            mistral_client=mistral_client,
        )

        # Prepare conversation history if available
        conversation_history = []
        if request.conversation_history:
            conversation_history = [
                ConversationMessage(role=msg.role, content=msg.content)
                for msg in request.conversation_history[-6:]  # Last 3 exchanges
            ]

        # Use the unified query service
        result = await query_service.query(
            query=request.query,
            conversation_history=conversation_history,
            top_k=request.top_k,
            rerank_k=request.rerank_k,
            threshold=request.threshold,
            alpha=request.alpha,
            lambda_param=request.lambda_param,
            use_mmr=request.use_mmr,
        )

        # Handle service errors
        if not result.success:
            logger.error(f"Query service error: {result.error}")
            raise HTTPException(
                status_code=500, detail=f"Query processing failed: {result.error}"
            )

        processing_time = time.time() - start_time
        logger.info(
            f"Query processed in {processing_time:.2f}s: '{request.query}' -> {len(result.citations)} citations"
        )

        return QueryResponse(
            answer=result.answer,
            citations=result.citations,
            insufficient_evidence=result.insufficient_evidence,
            intent=result.intent,
            debug=result.debug_info,
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/query/health")
async def query_health_check(
    db_manager: AsyncDatabaseManager = Depends(get_async_db_manager),
):
    """Health check for the query system."""
    try:
        # Check database connectivity
        async with aiosqlite.connect(db_manager.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM documents")
            total_docs_result = await cursor.fetchone()
            total_docs = total_docs_result[0] if total_docs_result else 0

            cursor = await conn.execute("SELECT COUNT(*) FROM chunks")
            total_chunks_result = await cursor.fetchone()
            total_chunks = total_chunks_result[0] if total_chunks_result else 0

        # Check Mistral API configuration
        mistral_configured = config.validate_mistral_config()

        # Test embedding if possible
        embedding_test_passed = False
        if mistral_configured and total_chunks > 0:
            try:
                mistral_client = get_mistral_client()
                test_embedding = await mistral_client.get_single_embedding_async("test")
                embedding_test_passed = len(test_embedding) > 0
            except Exception:
                pass

        return {
            "status": "healthy"
            if mistral_configured and total_docs > 0
            else "degraded",
            "database_connected": True,
            "mistral_api_configured": mistral_configured,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "embedding_test_passed": embedding_test_passed,
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database_connected": False,
            "mistral_api_configured": False,
            "total_documents": 0,
            "total_chunks": 0,
            "error": str(e),
        }
