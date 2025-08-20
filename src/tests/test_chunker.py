"""Tests for the chunking system."""

from app.ingest.chunker import HierarchicalChunker
from app.ingest.chunker import chunk_text


class TestHierarchicalChunker:
    """Test cases for the HierarchicalChunker class."""

    def test_chunk_small_text(self):
        """Test chunking of text smaller than target size."""
        chunker = HierarchicalChunker(target_size=100, overlap_size=20)
        pages_text = [(1, "This is a short text that should not be chunked.")]

        chunks = chunker.chunk_pages_text(pages_text)

        assert len(chunks) == 1
        assert chunks[0].text == "This is a short text that should not be chunked."
        assert chunks[0].page == 1
        assert chunks[0].position == 0

    def test_chunk_large_text_with_overlap(self):
        """Test chunking of large text with proper overlap."""
        chunker = HierarchicalChunker(
            target_size=30, overlap_size=10, min_chunk_size=10
        )

        # Create a text that will definitely need chunking with no sentence boundaries
        long_text = "word " * 50  # About 250 characters, no sentence boundaries
        pages_text = [(1, long_text)]

        chunks = chunker.chunk_pages_text(pages_text)

        # Should create multiple chunks for such long text with small target size
        assert len(chunks) >= 2
        assert all(chunk.page == 1 for chunk in chunks)
        assert chunks[0].position == 0
        if len(chunks) > 1:
            assert chunks[1].position == 1

        # Check that chunks have content
        assert all(len(chunk.text) > 0 for chunk in chunks)

    def test_chunk_multiple_pages(self):
        """Test chunking across multiple pages."""
        chunker = HierarchicalChunker(target_size=100, overlap_size=20)
        pages_text = [
            (1, "This is page one content. " * 5),
            (2, "This is page two content. " * 5),
            (3, "This is page three content. " * 5),
        ]

        chunks = chunker.chunk_pages_text(pages_text)

        # Should have chunks from different pages
        page_numbers = [chunk.page for chunk in chunks]
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers

        # Positions should increment across pages
        positions = [chunk.position for chunk in chunks]
        assert positions == sorted(positions)

    def test_chunk_empty_text(self):
        """Test handling of empty or whitespace-only text."""
        chunker = HierarchicalChunker(target_size=100, overlap_size=20)

        # Empty pages should be skipped
        pages_text = [(1, ""), (2, "   \n\n   "), (3, "This has actual content.")]

        chunks = chunker.chunk_pages_text(pages_text)

        assert len(chunks) == 1
        assert chunks[0].page == 3
        assert "actual content" in chunks[0].text

    def test_min_chunk_size_respected(self):
        """Test that minimum chunk size is respected."""
        chunker = HierarchicalChunker(
            target_size=50, overlap_size=10, min_chunk_size=30
        )

        # Create text with some very short segments
        text = (
            "Short. "
            + "This is a longer sentence that should be chunked properly. " * 3
        )
        pages_text = [(1, text)]

        chunks = chunker.chunk_pages_text(pages_text)

        # All chunks should meet minimum size requirement (or be the remainder)
        for chunk in chunks[:-1]:  # Exclude last chunk which might be shorter
            assert len(chunk.text.strip()) >= chunker.min_chunk_size


def test_chunk_text_convenience_function():
    """Test the convenience function for chunking."""
    pages_text = [
        (1, "This is page one with some content. " * 10),
        (2, "This is page two with different content. " * 10),
    ]

    chunks = chunk_text(pages_text, target_size=100, overlap_size=20)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Check that all chunks have required keys
    required_keys = {"text", "page", "position"}
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert required_keys.issubset(chunk.keys())
        assert isinstance(chunk["text"], str)
        assert isinstance(chunk["page"], int)
        assert isinstance(chunk["position"], int)


def test_chunk_text_boundary_detection():
    """Test that chunking respects sentence boundaries when possible."""
    chunker = HierarchicalChunker(target_size=100, overlap_size=20)

    # Text with clear sentence boundaries
    text = "First sentence here. Second sentence follows. Third sentence is longer and contains more details. Fourth sentence wraps up."
    pages_text = [(1, text)]

    chunks = chunker.chunk_pages_text(pages_text)

    if len(chunks) > 1:
        # Check that chunks tend to end at sentence boundaries
        first_chunk = chunks[0].text.strip()
        # Should end with punctuation if possible
        assert first_chunk[-1] in ".!?"


def test_chunk_text_preserves_content():
    """Test that chunking preserves all content without loss."""
    chunker = HierarchicalChunker(target_size=80, overlap_size=15)

    original_text = "This is the original text that needs to be chunked. " * 5
    pages_text = [(1, original_text)]

    chunks = chunker.chunk_pages_text(pages_text)

    # Collect all unique text from chunks (accounting for overlap)
    all_text = "".join(chunk.text for chunk in chunks)

    # The total length should be greater than original (due to overlap)
    # but all original words should be present
    original_words = set(original_text.split())
    chunk_words = set(all_text.split())

    # All original words should be preserved
    assert original_words.issubset(chunk_words)
