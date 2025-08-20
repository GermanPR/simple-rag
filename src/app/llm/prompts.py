"""Prompt templates for different query intents and generation tasks."""

# Avoid circular import - define enum locally for backward compatibility
from enum import Enum
from typing import Any


class QueryIntent(Enum):
    """Supported query intent types."""

    SMALLTALK = "smalltalk"
    QA = "qa"
    STRUCTURED = "structured"
    COMPARISON = "comparison"
    SUMMARY = "summary"


class PromptTemplate:
    """Base class for prompt templates."""

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format the prompt template with provided arguments."""
        raise NotImplementedError


class QAPromptTemplate(PromptTemplate):
    """Standard Q&A prompt template."""

    def __init__(self):
        self.system_prompt = """You are a helpful assistant that answers questions based strictly on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. Include citations in the format [filename p.X] for each claim
3. If you cannot answer from the context, respond with "Insufficient evidence to answer this question"
4. Do not use your general knowledge - only use the provided context
5. Be concise and accurate"""

        self.user_template = """Context:
{context}

Question: {query}

Answer with citations in the format [filename p.X]:"""

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


class StructuredPromptTemplate(PromptTemplate):
    """Template for structured responses (lists, steps, etc.)."""

    def __init__(self):
        self.system_prompt = """
        You are a helpful assistant that provides structured answers based strictly on the provided context.

        IMPORTANT RULES:
        1. Answer ONLY based on the provided context
        2. Use bullet points, numbered lists, or structured format as appropriate
        3. Include citations in the format [filename p.X] for each point
        4. If you cannot answer from the context, respond with "Insufficient evidence to answer this question"
        5. Do not use your general knowledge - only use the provided context"""

        self.user_template = """
        Context: {context}

        Question: {query}

        Provide a structured answer with citations [filename p.X]:"""

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


class ComparisonPromptTemplate(PromptTemplate):
    """Template for comparison-focused responses."""

    def __init__(self):
        self.system_prompt = """
        You are a helpful assistant that provides comparative analysis based strictly on the provided context.

        IMPORTANT RULES:
        1. Answer ONLY based on the provided context
        2. Structure your comparison clearly with distinct points
        3. Include citations in the format [filename p.X] for each comparison point
        4. If you cannot make the comparison from the context, respond with "Insufficient evidence to answer this question"
        5. Do not use your general knowledge - only use the provided context"""

        self.user_template = """
        Context: {context}

        Comparison request: {query}

        Provide a structured comparison with citations [filename p.X]:"""

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


class SummaryPromptTemplate(PromptTemplate):
    """Template for summary responses."""

    def __init__(self):
        self.system_prompt = """
        You are a helpful assistant that provides concise summaries based strictly on the provided context.

        IMPORTANT RULES:
        1. Summarize ONLY based on the provided context
        2. Focus on the most important points
        3. Include citations in the format [filename p.X] for key points
        4. If you cannot summarize from the context, respond with "Insufficient evidence to answer this question"
        5. Do not use your general knowledge - only use the provided context"""

        self.user_template = """
        Context: {context}

        Summary request: {query}

        Provide a concise summary with citations [filename p.X]:"""

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


class PromptManager:
    """Manages prompt templates for different intents."""

    def __init__(self):
        # Support both enum and string intents for backward compatibility
        self.templates = {
            QueryIntent.QA: QAPromptTemplate(),
            QueryIntent.STRUCTURED: StructuredPromptTemplate(),
            QueryIntent.COMPARISON: ComparisonPromptTemplate(),
            QueryIntent.SUMMARY: SummaryPromptTemplate(),
            # String versions for new LLM-based intent detection
            "qa": QAPromptTemplate(),
            "structured": StructuredPromptTemplate(),
            "comparison": ComparisonPromptTemplate(),
            "summary": SummaryPromptTemplate(),
        }
        # Default to QA template for unknown intents
        self.default_template = QAPromptTemplate()

    def get_prompt(
        self, intent: QueryIntent | str, query: str, context: str
    ) -> list[dict[str, str]]:
        """
        Get formatted prompt messages for the given intent.

        Args:
            intent: Detected query intent (enum or string)
            query: User query
            context: Retrieved context text

        Returns:
            List of message dictionaries for chat completion
        """
        template = self.templates.get(intent, self.default_template)
        return template.format(query=query, context=context)

    def format_context(self, chunks_info: list[dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks_info: List of chunk dictionaries with text, filename, page

        Returns:
            Formatted context string
        """
        if not chunks_info:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks_info, 1):
            text = chunk.get("text", "")
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page", 0)

            # Limit chunk text length for context
            max_chunk_display_length = 500
            if len(text) > max_chunk_display_length:
                text = text[:497] + "..."

            context_parts.append(f"[{i}] From {filename} p.{page}: {text}")

        return "\n\n".join(context_parts)


# Intent detection prompt for LLM-based intent detection and query rewriting
INTENT_DETECTION_PROMPT = """You are an expert at analyzing user queries to determine their intent and optimize them for document retrieval and comprehension.

Your task is to:
1. Analyze the user's current query in the context of their conversation history
2. Detect the most appropriate intent from these categories:
   - "smalltalk": Greetings, thanks, casual conversation not related to document content
   - "qa": Direct questions seeking specific information
   - "structured": Requests for lists, steps, procedures, organized information
   - "comparison": Requests to compare, contrast, or find differences/similarities
   - "summary": Requests for overviews, summaries, or high-level explanations

3. Rewrite the query to be more effective for information retrieval by:
   - Expanding abbreviations and acronyms
   - Adding context from conversation history when relevant
   - Clarifying ambiguous references
   - Optimizing for keyword and semantic search
   - Making implicit requests explicit

4. Provide a confidence score (0.0-1.0) for your intent classification
5. Explain your reasoning briefly

Consider the conversation history to understand:
- What the user has been asking about
- Any implicit context or continuation of previous topics
- References to earlier parts of the conversation

Your output should be a structured response with the detected intent, optimized query, confidence score, and reasoning."""

# Global prompt manager instance
prompt_manager = PromptManager()
