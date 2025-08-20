"""PDF text extraction using pdfminer.six."""

import asyncio
import logging
import re
from io import BytesIO
from typing import Any

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams
from pdfminer.layout import LTTextContainer
from pdfminer.pdfpage import PDFPage

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts text from PDF files page by page."""

    def __init__(self):
        self.laparams = LAParams(
            word_margin=0.1,
            char_margin=2.0,
            all_texts=False,
        )

    def extract_text_by_page(self, pdf_content: bytes) -> list[tuple[int, str]]:
        """
        Extract text from PDF, returning list of (page_number, text) tuples.

        Args:
            pdf_content: Raw PDF file content as bytes

        Returns:
            List of (page_number, cleaned_text) tuples
        """
        pages_text = []

        try:
            pdf_file = BytesIO(pdf_content)

            # Extract text page by page
            for page_num, page_layout in enumerate(
                extract_pages(pdf_file, laparams=self.laparams), 1
            ):
                page_text = self._extract_text_from_layout(page_layout)
                cleaned_text = self._clean_text(page_text)

                if cleaned_text.strip():  # Only include pages with meaningful content
                    pages_text.append((page_num, cleaned_text))

        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise

        return pages_text

    def _extract_text_from_layout(self, page_layout) -> str:
        """Extract text from a page layout object."""
        text_elements = []

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text_elements.append(element.get_text())

        return "".join(text_elements)

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

        # Basic header/footer removal (simple heuristic)
        min_lines_for_cleanup = 3
        min_header_footer_length = 50

        lines = text.split("\n")
        if len(lines) > min_lines_for_cleanup:
            # Remove very short first/last lines that might be headers/footers
            if len(lines[0].strip()) < min_header_footer_length and not lines[
                0
            ].strip().endswith("."):
                lines = lines[1:]
            if (
                len(lines) > 0
                and len(lines[-1].strip()) < min_header_footer_length
                and not lines[-1].strip().endswith(".")
            ):
                lines = lines[:-1]

        return " ".join(lines).strip()

    def get_pdf_info(self, pdf_content: bytes) -> dict[str, Any]:
        """Get basic information about the PDF."""
        try:
            pdf_file = BytesIO(pdf_content)
            pages = list(PDFPage.get_pages(pdf_file))

            return {
                "total_pages": len(pages),
                "has_text": True,  # We'll assume text extraction worked if we got here
            }
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            return {"total_pages": 0, "has_text": False, "error": str(e)}


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
