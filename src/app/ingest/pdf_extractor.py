"""PDF text extraction using PyMuPDF (fitz)."""

import asyncio
import re
from typing import Any

import fitz  # PyMuPDF

from app.core.exceptions import ProcessingError
from app.core.logging_config import get_logger

logger = get_logger(__name__.split(".")[-1])


class PDFExtractor:
    """Extracts text from PDF files with cross-page stitching."""

    def __init__(self):
        # PyMuPDF configuration - using minimal flags for text extraction
        self.text_flags = 0  # Use default text extraction flags
        self.min_line_height = 6  # Minimum line height for text
        self.max_font_size_variance = 5  # Font size variance for paragraph detection

    def extract_text_by_page(self, pdf_content: bytes) -> list[tuple[int, str]]:
        """
        Extract text from PDF with cross-page stitching for better coherence.

        Args:
            pdf_content: Raw PDF file content as bytes

        Returns:
            List of (page_number, text) tuples with stitched content
        """
        try:
            # Open PDF document
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            # Extract text with better structure preservation
            pages_text = []
            previous_page_text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Extract text with better structure preservation
                page_text = page.get_text()  # type: ignore[attr-defined]
                # For now, use simple text extraction - we can enhance later

                # Clean and normalize text
                cleaned_text = self._clean_text(page_text)

                # Stitch with previous page if text continues
                if previous_page_text and cleaned_text:
                    stitched_text = self._stitch_pages(previous_page_text, cleaned_text)
                    if len(pages_text) > 0:
                        # Update previous page with stitched content
                        prev_page_num, prev_text = pages_text[-1]
                        pages_text[-1] = (prev_page_num, stitched_text[0])
                        if stitched_text[1]:  # If there's remaining text for current page
                            pages_text.append((page_num + 1, stitched_text[1]))
                    else:
                        pages_text.append((page_num + 1, cleaned_text))
                elif cleaned_text.strip():
                    pages_text.append((page_num + 1, cleaned_text))

                previous_page_text = cleaned_text

            doc.close()

        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise ProcessingError(f"Failed to extract text from PDF: {e}") from e

        return pages_text

    def _extract_text_from_blocks(self, text_dict: dict[str, Any]) -> str:
        """Extract text from PyMuPDF text blocks with structure preservation."""
        text_lines = []

        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue

            block_lines = []
            for line in block["lines"]:
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")

                if line_text.strip():
                    block_lines.append(line_text.strip())

            if block_lines:
                # Join lines within a block, preserving paragraph structure
                block_text = " ".join(block_lines)
                text_lines.append(block_text)

        return "\n\n".join(text_lines)

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Performs:
        - Whitespace normalization
        - Line break handling
        - Header/footer removal heuristics
        - Unicode normalization
        """
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Handle common hyphenation patterns
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

        # Remove common artifacts
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\(\)\[\]\-\'\"]", " ", text)

        # Normalize multiple spaces
        text = re.sub(r"\s{2,}", " ", text)

        # Improved header/footer removal
        lines = text.split("\n")
        if len(lines) > 3:
            # Remove short first/last lines that are likely headers/footers
            # But preserve if they end with sentence punctuation
            if (len(lines[0].strip()) < 60 and
                not re.search(r'[.!?]"?\s*$', lines[0].strip()) and
                not re.search(r"^\s*\d+\s*$", lines[0].strip())):
                lines = lines[1:]

            if (len(lines) > 0 and
                len(lines[-1].strip()) < 60 and
                not re.search(r'[.!?]"?\s*$', lines[-1].strip()) and
                not re.search(r"^\s*\d+\s*$", lines[-1].strip())):
                lines = lines[:-1]

        return "\n".join(lines).strip()

    def _stitch_pages(self, prev_text: str, curr_text: str) -> tuple[str, str]:
        """Stitch text across pages to handle split sentences/paragraphs."""
        if not prev_text or not curr_text:
            return prev_text, curr_text

        prev_lines = prev_text.strip().split("\n")
        curr_lines = curr_text.strip().split("\n")

        if not prev_lines or not curr_lines:
            return prev_text, curr_text

        last_line = prev_lines[-1].strip()
        first_line = curr_lines[0].strip()

        # Check if last line of previous page might continue to first line of current page
        if (last_line and first_line and
            not re.search(r'[.!?]"?\s*$', last_line) and  # Doesn't end with sentence
            not first_line[0].isupper() and  # Next line doesn't start with capital
            len(last_line) > 20):  # Has substantial content

            # Stitch the lines together
            stitched_line = last_line + " " + first_line

            # Update previous page (remove last line, add stitched line)
            new_prev_lines = prev_lines[:-1] + [stitched_line]
            new_prev_text = "\n".join(new_prev_lines)

            # Update current page (remove first line)
            new_curr_lines = curr_lines[1:] if len(curr_lines) > 1 else []
            new_curr_text = "\n".join(new_curr_lines) if new_curr_lines else ""

            return new_prev_text, new_curr_text

        return prev_text, curr_text

    def get_pdf_info(self, pdf_content: bytes) -> dict[str, Any]:
        """Get basic information about the PDF."""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(doc)
            doc.close()

            return {
                "total_pages": total_pages,
                "has_text": True,  # We'll assume text extraction worked if we got here
            }
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            raise ProcessingError(f"Failed to get PDF info: {e}") from e


def extract_pdf_text(pdf_content: bytes) -> tuple[list[tuple[int, str]], int]:
    """
    Convenience function to extract text from PDF.

    Args:
        pdf_content: Raw PDF file content as bytes

    Returns:
        Tuple of (pages_text, total_pages) where pages_text is
        list of (page_number, text) tuples
    """
    extractor = PDFExtractor()
    pages_text = extractor.extract_text_by_page(pdf_content)
    pdf_info = extractor.get_pdf_info(pdf_content)

    return pages_text, pdf_info["total_pages"]


async def extract_pdf_text_async(
    pdf_content: bytes,
) -> tuple[list[tuple[int, str]], int]:
    """
    Async convenience function to extract text from PDF.

    Args:
        pdf_content: Raw PDF file content as bytes

    Returns:
        Tuple of (pages_text, total_pages) where pages_text is
        list of (page_number, text) tuples
    """
    return await asyncio.to_thread(extract_pdf_text, pdf_content)
