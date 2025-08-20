"""Unified RAG Query Processing Service.

This module provides a centralized service for handling RAG query processing
across different interfaces (API, CLI, UI). It encapsulates the entire pipeline
from intent detection to answer generation and post-processing.
"""

from typing import Any
from typing import final

from app.core.config import config
from app.core.exceptions import APIError
from app.core.exceptions import GenerationError
from app.core.exceptions import RetrievalError
from app.core.logging_config import get_logger
from app.core.models import CitationInfo
from app.core.models import ConversationMessageModel
from app.core.models import DebugInfo
from app.core.models import QueryServiceResult
from app.llm.mistral_client import EmbeddingManager
from app.llm.mistral_client import MistralClient
from app.llm.mistral_client import get_async_embedding_manager
from app.llm.prompts import prompt_manager
from app.logic.intent import ConversationMessage
from app.logic.intent import LLMIntentDetector
from app.logic.postprocess import answer_postprocessor
from app.retriever.fusion import AsyncHybridRetriever
from app.retriever.fusion import HybridRetriever
from app.retriever.index import AsyncDatabaseManager
from app.retriever.index import DatabaseManager
from app.retriever.keyword import AsyncKeywordSearcher
from app.retriever.keyword import KeywordSearcher
from app.retriever.semantic import AsyncSemanticSearcher
from app.retriever.semantic import SemanticSearcher

logger = get_logger(__name__.split(".")[-1])


@final
class RAGQueryService:
    """Unified service for RAG query processing.

    This service encapsulates the entire RAG pipeline:
    1. Intent detection and query rewriting
    2. Hybrid search (TF-IDF + semantic) with MMR diversification
    3. Evidence threshold validation
    4. Answer generation using LLM
    5. Post-processing with citation extraction

    It can be used synchronously by Streamlit UI and CLI, or asynchronously by FastAPI.
    """

    def __init__(
        self,
        db_path: str | None = None,
        mistral_client: MistralClient | None = None,
        db_manager: DatabaseManager | None = None,
    ):
        """Initialize the RAG query service.

        Args:
            db_path: Path to SQLite database (defaults to config.DB_PATH)
            mistral_client: Pre-initialized Mistral client (optional)
            db_manager: Pre-initialized database manager (optional)
        """
        self.db_path = db_path or config.DB_PATH
        self.mistral_client = mistral_client
        self.db_manager = db_manager

        # These will be initialized lazily
        self._embedding_manager = None
        self._keyword_searcher = None
        self._semantic_searcher = None
        self._intent_detector = None

    def _ensure_components_initialized(self):
        """Ensure all components are initialized."""
        if self.mistral_client is None:
            self.mistral_client = MistralClient()

        if self.db_manager is None:
            self.db_manager = DatabaseManager(self.db_path)

        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager(
                self.db_manager, self.mistral_client
            )

        if self._keyword_searcher is None:
            self._keyword_searcher = KeywordSearcher(self.db_manager)

        if self._semantic_searcher is None:
            self._semantic_searcher = SemanticSearcher(self.db_manager)

        if self._intent_detector is None:
            self._intent_detector = LLMIntentDetector(self.mistral_client)

        # Ensure all components are not None after initialization
        assert self._intent_detector is not None
        assert self._embedding_manager is not None

    def query(
        self,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
        top_k: int = 8,
        rerank_k: int = 20,
        threshold: float = 0.18,
        alpha: float | None = None,
        lambda_param: float | None = None,
        use_mmr: bool = True,
    ) -> QueryServiceResult:
        """Process a RAG query and return the result.

        Args:
            query: The user's question
            conversation_history: Optional conversation context
            top_k: Number of final results to return
            rerank_k: Number of candidates for reranking
            threshold: Minimum semantic similarity threshold
            alpha: Semantic vs keyword weight (0-1, defaults to config)
            lambda_param: MMR relevance vs diversity weight (0-1, defaults to config)
            use_mmr: Whether to apply MMR diversification

        Returns:
            QueryServiceResult containing answer, citations, and metadata
        """
        try:
            # Ensure components are initialized
            self._ensure_components_initialized()

            # Set default parameters
            if alpha is None:
                alpha = config.DEFAULT_ALPHA
            if lambda_param is None:
                lambda_param = config.DEFAULT_LAMBDA

            # Prepare conversation history
            if conversation_history is None:
                conversation_history = []
            else:
                # Limit to last 6 messages (3 exchanges)
                conversation_history = conversation_history[-6:]

            # Detect intent and rewrite query
            intent_result = self._intent_detector.detect_intent_and_rewrite(
                query, conversation_history
            )

            detected_intent = intent_result.intent
            processed_query = intent_result.rewritten_query

            # Handle smalltalk separately
            if detected_intent == "smalltalk":
                return QueryServiceResult(
                    answer=self._intent_detector.get_response_template(detected_intent),
                    citations=[],
                    insufficient_evidence=False,
                    intent=detected_intent,
                )

            # Create hybrid retriever
            hybrid_retriever = HybridRetriever(
                db_manager=self.db_manager,
                keyword_searcher=self._keyword_searcher,
                semantic_searcher=self._semantic_searcher,
                alpha=alpha,
                lambda_param=lambda_param,
            )

            # Generate query embedding
            query_embedding = self._embedding_manager.embed_query(processed_query)

            # Perform hybrid retrieval
            retrieved_results = hybrid_retriever.retrieve(
                query=processed_query,
                query_embedding=query_embedding,
                top_k=top_k,
                rerank_k=rerank_k,
                threshold=threshold,
                use_mmr=use_mmr,
            )

            # Check for insufficient evidence
            if not retrieved_results:
                debug_info = DebugInfo(
                    semantic_top1=0.0,
                    alpha=alpha,
                    lambda_param=lambda_param,
                    total_chunks=0,
                    keyword_results=0,
                    semantic_results=0,
                    fused_results=0,
                )
                return QueryServiceResult(
                    answer="Insufficient evidence to answer this question.",
                    citations=[],
                    insufficient_evidence=True,
                    intent=detected_intent,
                    debug_info=debug_info,
                )

            # Prepare context for generation
            chunks_info = [chunk_info for _, _, chunk_info in retrieved_results]
            context = prompt_manager.format_context(chunks_info)

            # Generate answer using appropriate prompt template
            messages = prompt_manager.get_prompt(detected_intent, query, context)

            logger.info(f"Messages: {str(messages)}")

            raw_answer = self.mistral_client.chat_completion(
                messages=messages, temperature=0.1, max_tokens=1000
            )
            logger.info(f"Raw answer: {raw_answer}")

            # Post-process answer
            processed_answer, citations, insufficient_evidence = (
                answer_postprocessor.process_answer(raw_answer, chunks_info)
            )
            logger.info(f"Processed answer: {processed_answer}\n\n")
            logger.info(f"Citations: {str(citations)}\n\n")
            logger.info(f"Infufficient evidence: {insufficient_evidence}\n\n")

            # Convert citations to the proper format
            citation_models = [
                CitationInfo(
                    chunk_id=c.chunk_id,
                    filename=c.filename,
                    page=c.page,
                    snippet=c.snippet,
                )
                for c in citations
            ]

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

            return QueryServiceResult(
                answer=processed_answer,
                citations=citation_models,
                insufficient_evidence=insufficient_evidence,
                intent=detected_intent,
                debug_info=debug_info,
            )

        except (RetrievalError, GenerationError, APIError) as e:
            logger.error(f"RAG system error in query processing: {e}")
            return QueryServiceResult(
                answer="I encountered an error while processing your question. Please try again.",
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error in RAG query processing: {e}")
            return QueryServiceResult(
                answer="An unexpected error occurred while processing your question.",
                citations=[],
                insufficient_evidence=True,
                intent="qa",
                error=str(e),
            )

    @staticmethod
    def convert_conversation_history(
        history: list[ConversationMessageModel] | list[dict[str, Any]] | None,
    ) -> list[ConversationMessage]:
        """Convert various conversation history formats to ConversationMessage objects.

        Args:
            history: List of conversation messages in various formats

        Returns:
            List of ConversationMessage objects
        """
        if not history:
            return []

        result = []
        for msg in history:
            if isinstance(msg, ConversationMessageModel):
                result.append(ConversationMessage(role=msg.role, content=msg.content))
            elif isinstance(msg, dict):
                result.append(
                    ConversationMessage(
                        role=msg.get("role", "user"), content=msg.get("content", "")
                    )
                )
            else:
                # Assume it's already a ConversationMessage
                result.append(msg)

        return result


class AsyncRAGQueryService:
    """Async version of RAGQueryService for FastAPI endpoints.

    This service provides the same functionality as RAGQueryService but uses
    async components for better performance in async web frameworks like FastAPI.
    """

    def __init__(
        self,
        async_db_manager: AsyncDatabaseManager,
        mistral_client: MistralClient | None = None,
    ):
        """Initialize the async RAG query service.

        Args:
            async_db_manager: Pre-initialized async database manager
            mistral_client: Pre-initialized Mistral client (optional)
        """
        self.async_db_manager = async_db_manager
        self.mistral_client = mistral_client

        # These will be initialized lazily
        self._async_embedding_manager = None
        self._async_keyword_searcher = None
        self._async_semantic_searcher = None
        self._intent_detector = None

    async def _ensure_components_initialized(self):
        """Ensure all async components are initialized."""
        if self.mistral_client is None:
            self.mistral_client = MistralClient()

        if self._async_embedding_manager is None:
            self._async_embedding_manager = get_async_embedding_manager(
                self.async_db_manager
            )

        if self._async_keyword_searcher is None:
            self._async_keyword_searcher = AsyncKeywordSearcher(self.async_db_manager)

        if self._async_semantic_searcher is None:
            self._async_semantic_searcher = AsyncSemanticSearcher(self.async_db_manager)

        if self._intent_detector is None:
            self._intent_detector = LLMIntentDetector(self.mistral_client)

        # Ensure all components are not None after initialization
        assert self._intent_detector is not None
        assert self._async_embedding_manager is not None

    async def query(
        self,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
        top_k: int = 8,
        rerank_k: int = 20,
        threshold: float = 0.18,
        alpha: float | None = None,
        lambda_param: float | None = None,
        use_mmr: bool = True,
    ) -> QueryServiceResult:
        """Process a RAG query asynchronously and return the result.

        Args:
            query: The user's question
            conversation_history: Optional conversation context
            top_k: Number of final results to return
            rerank_k: Number of candidates for reranking
            threshold: Minimum semantic similarity threshold
            alpha: Semantic vs keyword weight (0-1, defaults to config)
            lambda_param: MMR relevance vs diversity weight (0-1, defaults to config)
            use_mmr: Whether to apply MMR diversification

        Returns:
            QueryServiceResult containing answer, citations, and metadata
        """
        try:
            # Ensure components are initialized
            await self._ensure_components_initialized()

            # Set default parameters
            if alpha is None:
                alpha = config.DEFAULT_ALPHA
            if lambda_param is None:
                lambda_param = config.DEFAULT_LAMBDA

            # Prepare conversation history
            if conversation_history is None:
                conversation_history = []
            else:
                # Limit to last 6 messages (3 exchanges)
                conversation_history = conversation_history[-6:]

            # Detect intent and rewrite query
            intent_result = self._intent_detector.detect_intent_and_rewrite(
                query, conversation_history
            )

            detected_intent = intent_result.intent
            processed_query = intent_result.rewritten_query

            # Handle smalltalk separately
            if detected_intent == "smalltalk":
                return QueryServiceResult(
                    answer=self._intent_detector.get_response_template(detected_intent),
                    citations=[],
                    insufficient_evidence=False,
                    intent=detected_intent,
                )

            # Create hybrid retriever
            hybrid_retriever = AsyncHybridRetriever(
                async_db_manager=self.async_db_manager,
                async_keyword_searcher=self._async_keyword_searcher,
                async_semantic_searcher=self._async_semantic_searcher,
                alpha=alpha,
                lambda_param=lambda_param,
            )

            # Generate query embedding
            query_embedding = await self._async_embedding_manager.embed_query(
                processed_query
            )

            # Perform hybrid retrieval
            retrieved_results = await hybrid_retriever.retrieve(
                query=processed_query,
                query_embedding=query_embedding,
                top_k=top_k,
                rerank_k=rerank_k,
                threshold=threshold,
                use_mmr=use_mmr,
            )

            # Check for insufficient evidence
            if not retrieved_results:
                debug_info = DebugInfo(
                    semantic_top1=0.0,
                    alpha=alpha,
                    lambda_param=lambda_param,
                    total_chunks=0,
                    keyword_results=0,
                    semantic_results=0,
                    fused_results=0,
                )
                return QueryServiceResult(
                    answer="Insufficient evidence to answer this question.",
                    citations=[],
                    insufficient_evidence=True,
                    intent=detected_intent,
                    debug_info=debug_info,
                )

            # Prepare context for generation
            chunks_info = [chunk_info for _, _, chunk_info in retrieved_results]
            context = prompt_manager.format_context(chunks_info)

            # Generate answer using appropriate prompt template
            messages = prompt_manager.get_prompt(detected_intent, query, context)

            raw_answer = await self.mistral_client.chat_completion_async(
                messages=messages, temperature=0.1, max_tokens=1000
            )

            # Post-process answer
            processed_answer, citations, insufficient_evidence = (
                answer_postprocessor.process_answer(raw_answer, chunks_info)
            )
            if insufficient_evidence:
                debug_info = DebugInfo(
                    semantic_top1=0.0,
                    alpha=alpha,
                    lambda_param=lambda_param,
                    total_chunks=0,
                    keyword_results=0,
                    semantic_results=0,
                    fused_results=0,
                )
                return QueryServiceResult(
                    answer="There has been an error in the reponse, this can be hallucination or lack of evidence. Please try asking differently",
                    citations=[],
                    insufficient_evidence=True,
                    intent=detected_intent,
                    debug_info=debug_info,
                )


            # Convert citations to the proper format
            citation_models = [
                CitationInfo(
                    chunk_id=c.chunk_id,
                    filename=c.filename,
                    page=c.page,
                    snippet=c.snippet,
                )
                for c in citations
            ]

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

            return QueryServiceResult(
                answer=processed_answer,
                citations=citation_models,
                insufficient_evidence=insufficient_evidence,
                intent=detected_intent,
                debug_info=debug_info,
            )

        except (RetrievalError, GenerationError, APIError) as e:
            logger.error(f"RAG system error in async query processing: {e}")
            return QueryServiceResult(
                answer="I encountered an error while processing your question. Please try again.",
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error in async RAG query processing: {e}")
            return QueryServiceResult(
                answer="An unexpected error occurred while processing your question.",
                citations=[],
                insufficient_evidence=True,
                intent="qa",
                error=str(e),
            )
