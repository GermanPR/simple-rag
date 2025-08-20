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
    Creates overlapping text chunks using hierarchical structure.

    Attempts to respect paragraph and heading boundaries while maintaining
    target chunk sizes with specified overlap.
    """

    def __init__(
        self,
        target_size: int = 1800,
        overlap_size: int = 200,
        min_chunk_size: int = 100,
    ):
        self.target_size = target_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

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

            # Try to end at a good boundary
            chunk_text, actual_end = self._find_chunk_boundary(
                text, current_pos, chunk_end
            )

            # Ensure we always make some progress
            if actual_end <= current_pos:
                actual_end = min(current_pos + self.min_chunk_size, len(text))
                chunk_text = text[current_pos:actual_end]

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

            # Calculate next start position with overlap
            if actual_end >= len(text):
                break

            # Move forward, but include overlap
            overlap_start = max(current_pos, actual_end - self.overlap_size)
            new_pos = self._find_overlap_start(text, overlap_start, actual_end)

            # Ensure we make progress
            if new_pos <= current_pos:
                new_pos = current_pos + 1

            current_pos = new_pos

        return chunks

    def _find_chunk_boundary(
        self, text: str, start: int, target_end: int
    ) -> tuple[str, int]:
        """
        Find the best boundary for ending a chunk.

        Priority:
        1. Paragraph break
        2. Sentence boundary
        3. Word boundary
        4. Character boundary (fallback)
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
            # Use the last paragraph break
            match = paragraph_matches[-1]
            actual_end = start + (len(chunk_text) - len(last_portion)) + match.end()
            return text[start:actual_end], actual_end

        # Look for sentence boundaries in the last 150 chars
        search_back = min(150, len(chunk_text))
        search_text = chunk_text[-search_back:]

        sentence_matches = list(re.finditer(self.sentence_end_pattern, search_text))
        if sentence_matches:
            # Use the last sentence boundary
            match = sentence_matches[-1]
            actual_end = start + (len(chunk_text) - search_back) + match.end()
            return text[start:actual_end], actual_end

        # Look for word boundary
        if target_end < len(text):
            # Search backwards for a word boundary
            for i in range(min(50, target_end - start)):
                if text[target_end - i - 1].isspace():
                    actual_end = target_end - i
                    return text[start:actual_end], actual_end

        # Fallback to character boundary
        return chunk_text, target_end

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
    pages_text: list[tuple[int, str]], target_size: int = 1800, overlap_size: int = 200
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
    pages_text: list[tuple[int, str]], target_size: int = 1800, overlap_size: int = 200
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
