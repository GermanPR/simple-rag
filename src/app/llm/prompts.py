"""Prompt templates for different query intents and generation tasks."""

# Avoid circular import - define enum locally for backward compatibility
from enum import Enum
from typing import Any
from typing import final

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence to answer this question"

# Base system prompt with common rules
BASE_SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions based strictly on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. Include citations in the format [<filename> p.X] for each claim (e.g. [Organic_Law.pdf p.45])
3. If you cannot answer from the context, respond with {INSUFFICIENT_EVIDENCE_MESSAGE}
4. Do not use your general knowledge - only use the provided context
5. Be concise and accurate


How NOT TO WRITE CITATIONS: 
- ... [<filename> p.5] (actually using "<filename>" without replacing by the actual file name)
- ... [<filename>Real name.pdf p.6] (using <filename> without replacing but adding after)
- ... [Lord of the rings.pdf p.52, p.54, p.56] (putting multiple pages together from same source)

How TO ACTUALLY WRITE CITATIONS: 
- Input: From (filename="Lord of the rings.pdf" page="p.52"): the main character killed his father...
         From (filename="Lord of the rings.pdf" page="p.54"): his father was known by the name of john...
- Citation: the main character killed john [Lord of the rings.pdf p.52] [Lord of the rings.pdf p.54]
"""

# Standard user template
STANDARD_USER_TEMPLATE = """Context:
{context}

Question: {query}

Answer with citations in the format [<filename> p.X] (e.g. [Organic_Law.pdf p.45]):"""


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


@final
class QAPromptTemplate(PromptTemplate):
    """Standard Q&A prompt template."""

    def __init__(self):
        # Use base prompt as-is for standard Q&A
        self.system_prompt = BASE_SYSTEM_PROMPT
        self.user_template = STANDARD_USER_TEMPLATE

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


@final
class StructuredPromptTemplate(PromptTemplate):
    """Template for structured responses (lists, steps, etc.)."""

    def __init__(self):
        # Base prompt + structured formatting guidelines
        self.system_prompt = (
            BASE_SYSTEM_PROMPT
            + "\n\nStyle Guidelines: Use bullet points, numbered lists, or structured format as appropriate for each point."
        )
        self.user_template = STANDARD_USER_TEMPLATE.replace(
            "Answer with citations in the format [<filename> p.X]",
            "Provide a structured answer with citations [<filename> p.X]",
        )

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


@final
class ComparisonPromptTemplate(PromptTemplate):
    """Template for comparison-focused responses."""

    def __init__(self):
        # Base prompt + comparison-specific guidelines
        self.system_prompt = (
            BASE_SYSTEM_PROMPT
            + "\n\nStyle Guidelines: Structure your comparison clearly with distinct points for each item being compared."
        )
        self.user_template = STANDARD_USER_TEMPLATE.replace(
            "Question: {query}", "Comparison request: {query}"
        ).replace(
            "Answer with citations in the format [<filename> p.X]",
            "Provide a structured comparison with citations [<filename> p.X]",
        )

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


@final
class SummaryPromptTemplate(PromptTemplate):
    """Template for summary responses."""

    def __init__(self):
        # Base prompt + summary-specific guidelines
        self.system_prompt = (
            BASE_SYSTEM_PROMPT
            + "\n\nStyle Guidelines: Focus on the most important points and provide a concise overview."
        )
        self.user_template = STANDARD_USER_TEMPLATE.replace(
            "Question: {query}", "Summary request: {query}"
        ).replace(
            "Answer with citations in the format [<filename> p.X]",
            "Provide a concise summary with citations [<filename> p.X]",
        )

    def format(self, query: str, context: str, **kwargs) -> list[dict[str, str]]:
        """Format as messages for chat completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_template.format(query=query, context=context),
            },
        ]


@final
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

            context_parts.append(f"[{i}] From (filename='{filename}' page='p.{page}'): {text}")

        return "\n\n".join(context_parts)


# Intent detection prompt for LLM-based intent detection and query rewriting
INTENT_DETECTION_PROMPT = """You are an expert at analyzing user queries to determine their intent
and optimize them for document retrieval and comprehension.

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

# Hallucination detection prompt template
HALLUCINATION_DETECTION_PROMPT_TEMPLATE = """You are an expert fact-checker. Your task is to determine if a generated answer contains hallucinations or false information that cannot be supported by the provided context.

Context:
{context}

Generated Answer:
{answer}

Instructions:
1. Check every claim, fact, and statement in the answer
2. Verify if each claim can be directly supported by the context
3. Look for:
   - Information that contradicts the context
   - Claims not present in the context
   - Misinterpretation of the context
   - Addition of external knowledge not in context

Respond with ONLY "HALLUCINATION" if you find any unsupported claims, or "GROUNDED" if all claims are properly supported by the context.

After your verdict, provide a brief explanation in one sentence."""

# PII detection prompt template
PII_DETECTION_PROMPT_TEMPLATE = """You are a privacy expert. Analyze the following text for any personally identifiable information (PII).

Text to analyze:
{answer}

Look for these types of PII:
- Names (first names, last names, full names)
- Email addresses
- Phone numbers
- Physical addresses
- Social security numbers
- Credit card numbers
- Driver's license numbers
- Passport numbers
- Financial account numbers
- Medical record numbers
- IP addresses
- Usernames that could identify individuals
- Dates of birth
- Any other information that could identify a specific person

Respond with ONLY "PII_DETECTED" if you find any PII, or "NO_PII" if the text is clean.

If PII is detected, list the types found (e.g., "name, email, phone") on the next line."""

# Global prompt manager instance
prompt_manager = PromptManager()
