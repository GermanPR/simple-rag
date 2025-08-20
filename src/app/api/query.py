"""FastAPI endpoint for query processing."""

import logging
import time

import aiosqlite
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from app.core.config import config
from app.core.models import DebugInfo
from app.core.models import QueryRequest
from app.core.models import QueryResponse
from app.llm.mistral_client import get_async_embedding_manager
from app.llm.mistral_client import get_mistral_client
from app.llm.prompts import prompt_manager
from app.logic.intent import ConversationMessage
from app.logic.intent import LLMIntentDetector
from app.logic.postprocess import answer_postprocessor
from app.retriever.fusion import AsyncHybridRetriever
from app.retriever.index import AsyncDatabaseManager
from app.retriever.keyword import AsyncKeywordSearcher
from app.retriever.semantic import AsyncSemanticSearcher

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
        # Initialize components
        mistral_client = get_mistral_client()
        embedding_manager = get_async_embedding_manager(db_manager)
        keyword_searcher = AsyncKeywordSearcher(db_manager)
        semantic_searcher = AsyncSemanticSearcher(db_manager)

        # Set parameters (use request values or defaults)
        alpha = request.alpha if request.alpha is not None else config.DEFAULT_ALPHA
        lambda_param = (
            request.lambda_param
            if request.lambda_param is not None
            else config.DEFAULT_LAMBDA
        )

        hybrid_retriever = AsyncHybridRetriever(
            async_db_manager=db_manager,
            async_keyword_searcher=keyword_searcher,
            async_semantic_searcher=semantic_searcher,
            alpha=alpha,
            lambda_param=lambda_param,
        )

        # LLM-based intent detection and query rewriting
        llm_intent_detector = LLMIntentDetector(mistral_client)

        # Prepare conversation history if available
        conversation_history = []
        if request.conversation_history:
            conversation_history = [
                ConversationMessage(role=msg.role, content=msg.content)
                for msg in request.conversation_history[-6:]  # Last 3 exchanges
            ]

        # Detect intent and rewrite query with conversation context
        intent_result = llm_intent_detector.detect_intent_and_rewrite(
            request.query, conversation_history
        )

        detected_intent = intent_result.intent
        processed_query = intent_result.rewritten_query

        # Handle smalltalk separately
        if detected_intent == "smalltalk":
            return QueryResponse(
                answer=llm_intent_detector.get_response_template(detected_intent),
                citations=[],
                insufficient_evidence=False,
                intent=detected_intent,
                success=True,
            )

        # Generate query embedding
        try:
            query_embedding = await embedding_manager.embed_query(processed_query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to process query embedding"
            ) from e

        # Perform hybrid retrieval
        retrieved_results = await hybrid_retriever.retrieve(
            query=processed_query,
            query_embedding=query_embedding,
            top_k=request.top_k,
            rerank_k=request.rerank_k,
            threshold=request.threshold,
            use_mmr=request.use_mmr,
        )

        # Check for insufficient evidence
        if not retrieved_results:
            return QueryResponse(
                answer="Insufficient evidence to answer this question.",
                citations=[],
                insufficient_evidence=True,
                intent=detected_intent,
                debug=DebugInfo(
                    semantic_top1=0.0,
                    alpha=alpha,
                    lambda_param=lambda_param,
                    total_chunks=0,
                    keyword_results=0,
                    semantic_results=0,
                    fused_results=0,
                ),
                success=True,
            )

        # Prepare context for generation
        chunks_info = [chunk_info for _, _, chunk_info in retrieved_results]
        context = prompt_manager.format_context(chunks_info)

        # Generate answer using appropriate prompt template
        messages = prompt_manager.get_prompt(detected_intent, request.query, context)

        try:
            raw_answer = await mistral_client.chat_completion_async(
                messages=messages, temperature=0.1, max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to generate answer"
            ) from e

        # Post-process answer
        processed_answer, citations, insufficient_evidence = (
            answer_postprocessor.process_answer(raw_answer, chunks_info)
        )

        # Prepare debug information
        debug_info = DebugInfo(
            semantic_top1=retrieved_results[0][1] if retrieved_results else 0.0,
            alpha=alpha,
            lambda_param=lambda_param,
            total_chunks=len(retrieved_results),
            keyword_results=len(retrieved_results),  # Simplified for now
            semantic_results=len(retrieved_results),  # Simplified for now
            fused_results=len(retrieved_results),
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Query processed in {processing_time:.2f}s: '{request.query}' -> {len(citations)} citations"
        )

        return QueryResponse(
            answer=processed_answer,
            citations=citations,
            insufficient_evidence=insufficient_evidence,
            intent=detected_intent,
            debug=debug_info,
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
