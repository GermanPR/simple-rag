"""Text chunking with hierarchical structure and overlap."""

import asyncio
import re
from dataclasses import dataclass
from typing import Any

# Constants
PARAGRAPH_SEARCH_WINDOW = 200


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    page: int
    position: int
    start_char: int
    end_char: int


class HierarchicalChunker:
    """
    Creates overlapping text chunks using hierarchical structure with adaptive overlap.

    Attempts to respect paragraph and heading boundaries while maintaining
    target chunk sizes with boundary-aware overlap reduction.
    """

    def __init__(
        self,
        target_size: int = 1800,
        overlap_size: int = 100,
        min_chunk_size: int = 100,
        adaptive_overlap: bool = True,
    ):
        self.target_size = target_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.adaptive_overlap = adaptive_overlap

        # Adaptive overlap settings
        self.min_overlap = max(20, overlap_size // 4)  # Minimum 20 chars or 25% of base
        self.boundary_reduction_factor = 0.5  # Reduce overlap by 50% at boundaries

        # Patterns for identifying structure
        self.heading_patterns = [
            r"^\s*(?:Chapter|Section|Part)\s+\d+",  # Chapter/Section headers
            r"^\s*\d+\.\s+[A-Z]",  # Numbered headings
            r"^\s*[A-Z][A-Z\s]{10,}$",  # ALL CAPS headings
            r"^\s*[A-Z][a-z\s]+:$",  # Title Case headings ending with :
        ]

        self.sentence_end_pattern = r"[.!?]+(?:\s|$)"
        self.paragraph_break_pattern = r"\n\s*\n"

    def chunk_pages_text(self, pages_text: list[tuple[int, str]]) -> list[Chunk]:
        """
        Chunk text from multiple pages, preserving page boundaries in metadata.

        Args:
            pages_text: List of (page_number, text) tuples

        Returns:
            List of Chunk objects with page and position metadata
        """
        all_chunks = []
        global_position = 0

        for page_num, page_text in pages_text:
            if not page_text.strip():
                continue

            page_chunks = self._chunk_single_page(page_text, page_num, global_position)
            all_chunks.extend(page_chunks)
            global_position += len(page_chunks)

        return all_chunks

    def _chunk_single_page(
        self, text: str, page_num: int, start_position: int
    ) -> list[Chunk]:
        """Chunk text from a single page."""
        if len(text) <= self.target_size:
            return [
                Chunk(
                    text=text,
                    page=page_num,
                    position=start_position,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        chunks = []
        current_pos = 0
        chunk_position = start_position

        while current_pos < len(text):
            chunk_end = min(current_pos + self.target_size, len(text))

            # Ensure we make progress even with small chunks
            if chunk_end <= current_pos + self.min_chunk_size and chunk_end < len(text):
                chunk_end = min(current_pos + self.min_chunk_size, len(text))

            # Try to end at a good boundary and get boundary quality
            chunk_text, actual_end, boundary_quality = self._find_chunk_boundary_with_quality(
                text, current_pos, chunk_end
            )

            # Ensure we always make some progress
            if actual_end <= current_pos:
                actual_end = min(current_pos + self.min_chunk_size, len(text))
                chunk_text = text[current_pos:actual_end]
                boundary_quality = 0.0  # Poor boundary

            # Only create chunk if it meets minimum size or it's the last piece
            if len(chunk_text.strip()) >= self.min_chunk_size or actual_end >= len(
                text
            ):
                chunks.append(
                    Chunk(
                        text=chunk_text.strip(),
                        page=page_num,
                        position=chunk_position,
                        start_char=current_pos,
                        end_char=actual_end,
                    )
                )
                chunk_position += 1

            # Calculate next start position with adaptive overlap
            if actual_end >= len(text):
                break

            # Calculate adaptive overlap based on boundary quality
            effective_overlap = self._calculate_adaptive_overlap(boundary_quality)
            overlap_start = max(current_pos, actual_end - effective_overlap)
            new_pos = self._find_overlap_start(text, overlap_start, actual_end)

            # Ensure we make progress
            if new_pos <= current_pos:
                new_pos = current_pos + 1

            current_pos = new_pos

        return chunks

    def _find_chunk_boundary_with_quality(
        self, text: str, start: int, target_end: int
    ) -> tuple[str, int, float]:
        """
        Find the best boundary for ending a chunk with quality scoring.

        Priority and Quality Scores:
        1. Paragraph break (quality: 1.0)
        2. Sentence boundary (quality: 0.8)
        3. Word boundary (quality: 0.4)
        4. Character boundary (quality: 0.0)
        
        Returns:
            tuple: (chunk_text, actual_end, boundary_quality)
        """
        chunk_text = text[start:target_end]

        # Look for paragraph breaks in the last portion
        last_portion = (
            chunk_text[-PARAGRAPH_SEARCH_WINDOW:]
            if len(chunk_text) > PARAGRAPH_SEARCH_WINDOW
            else chunk_text
        )
        paragraph_matches = list(
            re.finditer(self.paragraph_break_pattern, last_portion)
        )

        if paragraph_matches:
            # Use the last paragraph break - highest quality
            match = paragraph_matches[-1]
            actual_end = start + (len(chunk_text) - len(last_portion)) + match.end()
            return text[start:actual_end], actual_end, 1.0

        # Look for sentence boundaries in the last 150 chars
        search_back = min(150, len(chunk_text))
        search_text = chunk_text[-search_back:]

        sentence_matches = list(re.finditer(self.sentence_end_pattern, search_text))
        if sentence_matches:
            # Use the last sentence boundary - high quality
            match = sentence_matches[-1]
            actual_end = start + (len(chunk_text) - search_back) + match.end()
            return text[start:actual_end], actual_end, 0.8

        # Look for word boundary
        if target_end < len(text):
            # Search backwards for a word boundary
            for i in range(min(50, target_end - start)):
                if text[target_end - i - 1].isspace():
                    actual_end = target_end - i
                    return text[start:actual_end], actual_end, 0.4

        # Fallback to character boundary - lowest quality
        return chunk_text, target_end, 0.0

    def _calculate_adaptive_overlap(self, boundary_quality: float) -> int:
        """
        Calculate effective overlap size based on boundary quality.
        
        High-quality boundaries (paragraphs, sentences) get reduced overlap.
        Poor boundaries (character breaks) get full overlap.
        """
        if not self.adaptive_overlap:
            return self.overlap_size

        # Reduce overlap for high-quality boundaries
        reduction = boundary_quality * self.boundary_reduction_factor
        effective_overlap = int(self.overlap_size * (1 - reduction))

        # Ensure minimum overlap
        return max(self.min_overlap, effective_overlap)

    def _find_overlap_start(self, text: str, overlap_start: int, chunk_end: int) -> int:
        """
        Find a good starting point for overlap that doesn't break words.
        """
        if overlap_start >= chunk_end:
            return chunk_end

        # Try to start at a word boundary
        for i in range(min(30, chunk_end - overlap_start)):
            if text[overlap_start + i].isspace():
                return overlap_start + i + 1

        return overlap_start

    def _is_heading(self, text: str) -> bool:
        """Check if text appears to be a heading."""
        text = text.strip()
        if not text:
            return False

        for pattern in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False


def chunk_text(
    pages_text: list[tuple[int, str]], target_size: int = 1800, overlap_size: int = 100
) -> list[dict[str, Any]]:
    """
    Convenience function to chunk text and return as dictionaries.

    Args:
        pages_text: List of (page_number, text) tuples
        target_size: Target chunk size in characters
        overlap_size: Overlap size in characters

    Returns:
        List of chunk dictionaries with keys: text, page, position
    """
    chunker = HierarchicalChunker(target_size, overlap_size)
    chunks = chunker.chunk_pages_text(pages_text)

    return [
        {"text": chunk.text, "page": chunk.page, "position": chunk.position}
        for chunk in chunks
    ]


async def chunk_text_async(
    pages_text: list[tuple[int, str]], target_size: int = 1800, overlap_size: int = 100
) -> list[dict[str, Any]]:
    """
    Async convenience function to chunk text and return as dictionaries.

    Args:
        pages_text: List of (page_number, text) tuples
        target_size: Target chunk size in characters
        overlap_size: Overlap size in characters

    Returns:
        List of chunk dictionaries with keys: text, page, position
    """
    return await asyncio.to_thread(chunk_text, pages_text, target_size, overlap_size)
