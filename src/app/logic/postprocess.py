"""Post-processing for citations, answer formatting, and validation."""

import re
from typing import Any

from app.core.models import CitationInfo


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
        # Build set of valid citations from available chunks
        valid_citations = set()
        for chunk in available_chunks:
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page", 1)
            valid_citations.add(f"{filename} p.{page}")

        # Find all citations in the answer
        found_citations = self.citation_pattern.findall(answer)
        all_valid = True

        for filename, page in found_citations:
            citation_key = f"{filename} p.{page}"
            if citation_key not in valid_citations:
                all_valid = False
                # Could remove invalid citation or flag it
                # For now, we'll keep it but note the validation failure

        return answer, all_valid

    def ensure_citations_present(
        self, answer: str, chunks_info: list[dict[str, Any]], min_citations: int = 1
    ) -> str:
        """
        Ensure the answer contains minimum number of citations.

        Args:
            answer: Generated answer
            chunks_info: Available chunks for citation
            min_citations: Minimum required citations

        Returns:
            Answer with citations added if necessary
        """
        existing_citations = len(self.citation_pattern.findall(answer))

        if existing_citations >= min_citations:
            return answer

        # Add citations from top chunks if missing
        needed_citations = min_citations - existing_citations
        citations_to_add = []

        for chunk in chunks_info[:needed_citations]:
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page", 1)
            citation = f"[{filename} p.{page}]"
            citations_to_add.append(citation)

        if citations_to_add:
            # Add citations at the end
            citations_text = " " + " ".join(citations_to_add)
            answer += citations_text

        return answer


class AnswerValidator:
    """Validates generated answers for quality and safety."""

    def __init__(self):
        # Keywords that might indicate hallucination or unsafe content
        self.warning_phrases = {
            "medical": [
                "diagnosis",
                "prescribe",
                "medication",
                "treatment",
                "medical advice",
            ],
            "legal": [
                "legal advice",
                "lawsuit",
                "attorney",
                "court case",
                "legal counsel",
            ],
            "financial": [
                "investment advice",
                "financial advice",
                "buy stocks",
                "invest in",
            ],
            "harmful": ["violence", "illegal", "dangerous", "harmful"],
        }

        self.insufficient_evidence_phrases = [
            "insufficient evidence",
            "cannot answer from the context",
            "not enough information",
            "unable to answer based on the provided context",
        ]

    def is_insufficient_evidence_response(self, answer: str) -> bool:
        """
        Check if the answer indicates insufficient evidence.

        Args:
            answer: Generated answer text

        Returns:
            True if answer indicates insufficient evidence
        """
        answer_lower = answer.lower()
        return any(
            phrase in answer_lower for phrase in self.insufficient_evidence_phrases
        )

    def check_for_warnings(self, answer: str) -> list[str]:
        """
        Check answer for potentially unsafe or concerning content.

        Args:
            answer: Generated answer text

        Returns:
            List of warning categories found
        """
        warnings = []
        answer_lower = answer.lower()

        for category, phrases in self.warning_phrases.items():
            if any(phrase in answer_lower for phrase in phrases):
                warnings.append(category)

        return warnings

    def add_disclaimers(self, answer: str, warnings: list[str]) -> str:
        """
        Add appropriate disclaimers based on detected warnings.

        Args:
            answer: Original answer
            warnings: List of warning categories

        Returns:
            Answer with disclaimers added
        """
        disclaimers = {
            "medical": "\n\n**Disclaimer:** This information is for educational purposes only and should not be considered medical advice. Consult with a healthcare professional for medical concerns.",
            "legal": "\n\n**Disclaimer:** This information is for educational purposes only and should not be considered legal advice. Consult with a qualified attorney for legal matters.",
            "financial": "\n\n**Disclaimer:** This information is for educational purposes only and should not be considered financial advice. Consult with a qualified financial advisor for investment decisions.",
        }

        modified_answer = answer
        for warning in warnings:
            if warning in disclaimers:
                modified_answer += disclaimers[warning]

        return modified_answer


class AnswerPostProcessor:
    """Main class for post-processing generated answers."""

    def __init__(self):
        self.citation_processor = CitationProcessor()
        self.answer_validator = AnswerValidator()

    def process_answer(
        self,
        answer: str,
        chunks_info: list[dict[str, Any]],
        add_disclaimers: bool = True,
    ) -> tuple[str, list[CitationInfo], bool]:
        """
        Post-process a generated answer.

        Args:
            answer: Raw generated answer
            chunks_info: Chunks used for context
            add_disclaimers: Whether to add safety disclaimers

        Returns:
            Tuple of (processed_answer, citations, insufficient_evidence)
        """
        # Check for insufficient evidence
        insufficient_evidence = self.answer_validator.is_insufficient_evidence_response(
            answer
        )

        if insufficient_evidence:
            return answer, [], True

        # Validate citations
        validated_answer, citations_valid = (
            self.citation_processor.validate_citations_in_answer(answer, chunks_info)
        )

        # Ensure minimum citations are present
        answer_with_citations = self.citation_processor.ensure_citations_present(
            validated_answer, chunks_info, min_citations=1
        )

        # Check for warnings and add disclaimers
        if add_disclaimers:
            warnings = self.answer_validator.check_for_warnings(answer_with_citations)
            if warnings:
                answer_with_citations = self.answer_validator.add_disclaimers(
                    answer_with_citations, warnings
                )

        # Extract citation information
        citations = self.citation_processor.extract_citations_from_chunks(chunks_info)

        return answer_with_citations, citations, False


# Global post-processor instance
answer_postprocessor = AnswerPostProcessor()
