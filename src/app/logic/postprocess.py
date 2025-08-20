"""Post-processing for citations, answer formatting, and validation."""

import re
from typing import Any
from typing import final

from app.core.exceptions import GenerationError
from app.core.logging_config import get_logger
from app.core.models import CitationInfo
from app.llm.mistral_client import MistralClient
from app.llm.prompts import HALLUCINATION_DETECTION_PROMPT_TEMPLATE
from app.llm.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from app.llm.prompts import PII_DETECTION_PROMPT_TEMPLATE

logger = get_logger(__name__.split(".")[-1])

# Standardized error messages
HALLUCINATION_ERROR_MESSAGE = "I apologize, but I cannot provide a reliable answer based on the available information. The response may contain inaccurate information that is not supported by the source documents."

PII_ERROR_MESSAGE = "I apologize, but I cannot provide this response as it may contain personally identifiable information that should not be shared."


@final
class CitationProcessor:
    """Processes and validates citations in generated answers."""

    def __init__(self):
        # Pattern to match citations in format [filename p.X]
        self.citation_pattern = re.compile(r"\[([^[\]]+)\s+p\.(\d+)\]")
        self.max_snippet_length = 150

    def extract_citations_from_chunks(
        self, chunks_info: list[dict[str, Any]]
    ) -> list[CitationInfo]:
        """
        Extract citation information from chunk data.

        Args:
            chunks_info: List of chunk dictionaries with metadata

        Returns:
            List of CitationInfo objects
        """
        citations = []

        for chunk in chunks_info:
            chunk_id = chunk.get("chunk_id")
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page", 1)
            text = chunk.get("text", "")

            # Create snippet (first N characters)
            snippet = text[: self.max_snippet_length]
            if len(text) > self.max_snippet_length:
                snippet += "..."

            citation = CitationInfo(
                chunk_id=chunk_id or 0, filename=filename, page=page, snippet=snippet
            )
            citations.append(citation)

        return citations

    def validate_citations_in_answer(
        self, answer: str, available_chunks: list[dict[str, Any]]
    ) -> tuple[str, bool]:
        """
        Validate that citations in the answer correspond to available chunks.

        Args:
            answer: Generated answer with citations
            available_chunks: List of chunks that were provided as context

        Returns:
            Tuple of (validated_answer, all_citations_valid)
        """
        # Build set of valid citations from available chunks (handle all possible formats)
        valid_citations = set()
        filename_to_page = {}
        for i, chunk in enumerate(available_chunks, 1):
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page", 1)
            # Add multiple citation formats that LLM might use:
            valid_citations.add(f"{filename} p.{page}")  # [filename.pdf p.5]
            # Store mapping for replacement
            if filename not in filename_to_page:
                filename_to_page[filename] = set()
            filename_to_page[filename].add(page)

        # Find all citations in the answer
        found_citations = self.citation_pattern.findall(answer)
        all_valid = True

        for citation_ref, page in found_citations:
            citation_key = f"{citation_ref} p.{page}"
            if citation_key not in valid_citations:
                all_valid = False
                logger.warning(
                    f"Invalid citation found: '{citation_key}' - not in valid set"
                )

        return answer, all_valid


@final
class HallucinationDetector:
    """LLM-based detector for hallucinations in generated answers."""

    def __init__(self, mistral_client: MistralClient | None = None):
        self.mistral_client = mistral_client or MistralClient()

    def check_hallucination(self, answer: str, context: str) -> tuple[bool, str]:
        """
        Check if the answer contains hallucinations based on the provided context.

        Args:
            answer: Generated answer to check
            context: Original context chunks used for generation

        Returns:
            Tuple of (is_hallucinated, reasoning)
        """
        if not answer.strip() or not context.strip():
            return False, "Empty answer or context"

        hallucination_prompt = HALLUCINATION_DETECTION_PROMPT_TEMPLATE.format(
            context=context, answer=answer
        )

        try:
            messages = [{"role": "user", "content": hallucination_prompt}]

            response = self.mistral_client.chat_completion(
                messages=messages, temperature=0.0, max_tokens=150
            )

            response_lower = response.lower().strip()
            is_hallucinated = "hallucination" in response_lower

            # Extract reasoning from response
            lines = response.strip().split("\n")
            MAX_LINES_FOR_SINGLE_RESPONSE = 2
            reasoning = (
                response
                if len(lines) <= MAX_LINES_FOR_SINGLE_RESPONSE
                else " ".join(lines[1:]).strip()
            )
            if not reasoning:
                reasoning = "LLM response analysis"

            logger.info(
                f"Hallucination check: {'DETECTED' if is_hallucinated else 'PASSED'} - {reasoning}"
            )

            return is_hallucinated, reasoning

        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            # Default to safe mode - assume hallucination if check fails
            raise GenerationError(f"Hallucination detection failed: {e}") from e


@final
class PIIDetector:
    """LLM-based detector for personally identifiable information (PII)."""

    def __init__(self, mistral_client: MistralClient | None = None):
        self.mistral_client = mistral_client or MistralClient()

    def check_pii(self, answer: str) -> tuple[bool, list[str]]:
        """
        Check if the answer contains personally identifiable information.

        Args:
            answer: Generated answer to check for PII

        Returns:
            Tuple of (contains_pii, list_of_pii_types_found)
        """
        if not answer.strip():
            return False, []

        pii_prompt = PII_DETECTION_PROMPT_TEMPLATE.format(answer=answer)

        try:
            messages = [{"role": "user", "content": pii_prompt}]

            response = self.mistral_client.chat_completion(
                messages=messages, temperature=0.0, max_tokens=100
            )

            response_lower = response.lower().strip()
            contains_pii = "pii_detected" in response_lower

            # Extract PII types if detected
            pii_types = []
            if contains_pii:
                lines = response.strip().split("\n")
                if len(lines) > 1:
                    # Parse the second line for PII types
                    types_line = lines[1].strip()
                    pii_types = [t.strip() for t in types_line.split(",") if t.strip()]

            logger.info(
                f"PII check: {'DETECTED' if contains_pii else 'PASSED'} - Types: {pii_types}"
            )

            return contains_pii, pii_types

        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            # Default to safe mode - assume PII if check fails
            raise GenerationError(f"PII detection failed: {e}") from e


@final
class AnswerPostProcessor:
    """Main class for post-processing generated answers."""

    def __init__(self, mistral_client: MistralClient | None = None):
        self.citation_processor = CitationProcessor()
        self.hallucination_detector = HallucinationDetector(mistral_client)
        self.pii_detector = PIIDetector(mistral_client)

    def process_answer(
        self,
        answer: str,
        chunks_info: list[dict[str, Any]],
        context: str | None = None,
        enable_hallucination_check: bool = True,
        enable_pii_check: bool = True,
    ) -> tuple[str, list[CitationInfo], bool]:
        """
        Post-process a generated answer with safety checks.

        Args:
            answer: Raw generated answer
            chunks_info: Chunks used for context
            context: Original context string used for generation
            enable_hallucination_check: Whether to run hallucination detection
            enable_pii_check: Whether to run PII detection

        Returns:
            Tuple of (processed_answer, citations, insufficient_evidence)
        """
        insufficient_evidence = INSUFFICIENT_EVIDENCE_MESSAGE in answer

        if insufficient_evidence:
            logger.info("Insufficient evidence phrase found")
            return answer, [], True

        # Run PII check first (faster and prevents exposure of sensitive data)
        if enable_pii_check:
            contains_pii, pii_types = self.pii_detector.check_pii(answer)
            if contains_pii:
                logger.warning(f"PII detected in answer. Types: {pii_types}")
                return PII_ERROR_MESSAGE, [], True

        # Run hallucination check if context is provided
        if enable_hallucination_check and context:
            is_hallucinated, reasoning = (
                self.hallucination_detector.check_hallucination(answer, context)
            )
            if is_hallucinated:
                logger.warning(
                    f"Hallucination detected in answer. Reasoning: {reasoning}"
                )
                return HALLUCINATION_ERROR_MESSAGE, [], True

        # Validate citations
        validated_answer, citations_valid = (
            self.citation_processor.validate_citations_in_answer(answer, chunks_info)
        )
        if not citations_valid:
            logger.info("Citations are not valid")
            return answer, [], True

        # Extract citation information
        citations = self.citation_processor.extract_citations_from_chunks(chunks_info)

        return validated_answer, citations, False


# Global post-processor instance
answer_postprocessor = AnswerPostProcessor()
