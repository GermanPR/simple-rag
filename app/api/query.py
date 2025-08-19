"""FastAPI endpoint for query processing."""

from fastapi import APIRouter, HTTPException, Depends
import logging
import time

from app.core.models import QueryRequest, QueryResponse, DebugInfo, ErrorResponse
from app.retriever.index import DatabaseManager
from app.retriever.keyword import KeywordSearcher
from app.retriever.semantic import SemanticSearcher
from app.retriever.fusion import HybridRetriever
from app.llm.mistral_client import get_mistral_client, get_embedding_manager
from app.logic.intent import IntentDetector, QueryRewriter
from app.llm.prompts import prompt_manager
from app.logic.postprocess import answer_postprocessor
from app.core.config import config

logger = logging.getLogger(__name__)

router = APIRouter()


def get_db_manager():
    """Dependency to get database manager."""
    return DatabaseManager(config.DB_PATH)


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db_manager: DatabaseManager = Depends(get_db_manager)
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
            detail="Mistral API key not configured. Please set MISTRAL_API_KEY environment variable."
        )
    
    try:
        # Initialize components
        mistral_client = get_mistral_client()
        embedding_manager = get_embedding_manager(db_manager)
        keyword_searcher = KeywordSearcher(db_manager)
        semantic_searcher = SemanticSearcher(db_manager)
        
        # Set parameters (use request values or defaults)
        alpha = request.alpha if request.alpha is not None else config.DEFAULT_ALPHA
        lambda_param = request.lambda_param if request.lambda_param is not None else config.DEFAULT_LAMBDA
        
        hybrid_retriever = HybridRetriever(
            db_manager=db_manager,
            keyword_searcher=keyword_searcher,
            semantic_searcher=semantic_searcher,
            alpha=alpha,
            lambda_param=lambda_param
        )
        
        # Intent detection and query rewriting
        intent_detector = IntentDetector()
        query_rewriter = QueryRewriter()
        
        detected_intent = intent_detector.detect_intent(request.query)
        
        # Handle smalltalk separately
        if detected_intent.value == "smalltalk":
            return QueryResponse(
                answer=intent_detector.get_response_template(detected_intent),
                citations=[],
                insufficient_evidence=False,
                intent=detected_intent.value,
                success=True
            )
        
        # Rewrite and normalize query
        processed_query = query_rewriter.rewrite_query(request.query, detected_intent)
        
        # Generate query embedding
        try:
            query_embedding = embedding_manager.embed_query(processed_query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise HTTPException(status_code=500, detail="Failed to process query embedding")
        
        # Perform hybrid retrieval
        retrieved_results = hybrid_retriever.retrieve(
            query=processed_query,
            query_embedding=query_embedding,
            top_k=request.top_k,
            rerank_k=request.rerank_k,
            threshold=request.threshold,
            use_mmr=request.use_mmr
        )
        
        # Check for insufficient evidence
        if not retrieved_results:
            return QueryResponse(
                answer="Insufficient evidence to answer this question.",
                citations=[],
                insufficient_evidence=True,
                intent=detected_intent.value,
                debug=DebugInfo(
                    semantic_top1=0.0,
                    alpha=alpha,
                    lambda_param=lambda_param,
                    total_chunks=0,
                    keyword_results=0,
                    semantic_results=0,
                    fused_results=0
                ),
                success=True
            )
        
        # Prepare context for generation
        chunks_info = [chunk_info for _, _, chunk_info in retrieved_results]
        context = prompt_manager.format_context(chunks_info)
        
        # Generate answer using appropriate prompt template
        messages = prompt_manager.get_prompt(detected_intent, request.query, context)
        
        try:
            raw_answer = mistral_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate answer")
        
        # Post-process answer
        processed_answer, citations, insufficient_evidence = answer_postprocessor.process_answer(
            raw_answer, chunks_info
        )
        
        # Prepare debug information
        debug_info = DebugInfo(
            semantic_top1=retrieved_results[0][1] if retrieved_results else 0.0,
            alpha=alpha,
            lambda_param=lambda_param,
            total_chunks=len(retrieved_results),
            keyword_results=len(retrieved_results),  # Simplified for now
            semantic_results=len(retrieved_results),  # Simplified for now
            fused_results=len(retrieved_results)
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s: '{request.query}' -> {len(citations)} citations")
        
        return QueryResponse(
            answer=processed_answer,
            citations=citations,
            insufficient_evidence=insufficient_evidence,
            intent=detected_intent.value,
            debug=debug_info,
            success=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/query/health")
async def query_health_check(db_manager: DatabaseManager = Depends(get_db_manager)):
    """Health check for the query system."""
    try:
        # Check database connectivity
        with db_manager.db_path as path:
            import sqlite3
            with sqlite3.connect(path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                total_docs = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")  
                total_chunks = cursor.fetchone()[0]
        
        # Check Mistral API configuration
        mistral_configured = config.validate_mistral_config()
        
        # Test embedding if possible
        embedding_test_passed = False
        if mistral_configured and total_chunks > 0:
            try:
                mistral_client = get_mistral_client()
                test_embedding = mistral_client.get_single_embedding("test")
                embedding_test_passed = len(test_embedding) > 0
            except:
                pass
        
        return {
            "status": "healthy" if mistral_configured and total_docs > 0 else "degraded",
            "database_connected": True,
            "mistral_api_configured": mistral_configured,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "embedding_test_passed": embedding_test_passed
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database_connected": False,
            "mistral_api_configured": False,
            "total_documents": 0,
            "total_chunks": 0,
            "error": str(e)
        }