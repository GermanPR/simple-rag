"""LLM-based intent detection and query rewriting."""

from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from app.core.logging_config import get_logger
from app.llm.mistral_client import MistralClient
from app.llm.prompts import INTENT_DETECTION_PROMPT

logger = get_logger(__name__.split(".")[-1])


QueryIntent = Literal["smalltalk", "qa", "structured", "comparison", "summary"]


class ConversationMessage(BaseModel):
    """Represents a single message in conversation history."""

    role: str = Field(description="Message role: 'user' or 'assistant'")
    content: str = Field(description="Message content")


class IntentDetectionResult(BaseModel):
    """Structured output for intent detection and query rewriting."""

    intent: QueryIntent = Field(
        description="Detected intent: smalltalk, qa, structured, comparison, or summary"
    )
    rewritten_query: str = Field(
        description="Optimized version of the user query for better retrieval and understanding"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 for the detected intent"
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was detected and how the query was rewritten"
    )


class LLMIntentDetector:
    """LLM-based intent detection and query rewriting with conversation history."""

    def __init__(self, mistral_client: MistralClient | None = None):
        self.mistral_client = mistral_client or MistralClient()

    def detect_intent_and_rewrite(
        self,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> IntentDetectionResult:
        """
        Detect intent and rewrite query using LLM with conversation context.

        Args:
            query: Current user query
            conversation_history: Last 3 user messages and responses

        Returns:
            IntentDetectionResult with intent, rewritten query, confidence, and reasoning
        """
        if not query.strip():
            return IntentDetectionResult(
                intent="smalltalk",
                rewritten_query="",
                confidence=1.0,
                reasoning="Empty query treated as smalltalk",
            )

        # Prepare conversation context
        context_str = ""
        if conversation_history:
            context_messages = []
            for msg in conversation_history[-6:]:  # Last 3 exchanges (6 messages)
                context_messages.append(f"{msg.role.title()}: {msg.content}")
            if context_messages:
                context_str = "\n".join(context_messages)

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": INTENT_DETECTION_PROMPT},
            {
                "role": "user",
                "content": f"""Current query: {query}

                Conversation history:
                {context_str if context_str else "No previous conversation"}

                Analyze the current query in the context of the conversation history.""",
            },
        ]

        try:
            # Use structured completion with the official Mistral AI client
            return self.mistral_client.structured_chat_completion(
                messages=messages,
                response_format=IntentDetectionResult,
                temperature=0.1,
                max_tokens=500,
            )

        except Exception as e:
            logger.error(f"Error in LLM intent detection: {e}")
            # Use fallback instead of raising to maintain functionality
            return self._fallback_intent_detection(query)

    def _fallback_intent_detection(self, query: str) -> IntentDetectionResult:
        """Fallback intent detection using simple heuristics."""
        query_lower = query.lower().strip()

        # Simple pattern matching as fallback
        if any(word in query_lower for word in ["hi", "hello", "thanks", "bye"]):
            intent = "smalltalk"
        elif any(
            word in query_lower for word in ["list", "steps", "enumerate", "bullet"]
        ):
            intent = "structured"
        elif any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            intent = "comparison"
        elif any(word in query_lower for word in ["summary", "summarize", "overview"]):
            intent = "summary"
        else:
            intent = "qa"

        return IntentDetectionResult(
            intent=intent,
            rewritten_query=query.strip(),
            confidence=0.7,
            reasoning=f"Fallback detection based on keywords for {intent}",
        )

    def get_response_template(self, intent: str) -> str:
        """
        Get the appropriate response template for an intent.

        Args:
            intent: Detected query intent string

        Returns:
            Template string for formatting responses
        """
        templates = {
            "smalltalk": "I'm here to help answer questions about your documents. Please ask me something about the content you've uploaded.",
            "qa": "Answer the question based only on the provided context. If you cannot answer from the context, say 'Insufficient evidence to answer this question.'",
            "structured": "Provide the answer in a structured format using bullet points or numbered lists. Base your answer only on the provided context.",
            "comparison": "Compare the items mentioned in the query based only on the provided context. Structure your comparison clearly.",
            "summary": "Provide a concise summary based only on the provided context. Focus on the key points.",
        }

        return templates.get(intent, templates["qa"])
