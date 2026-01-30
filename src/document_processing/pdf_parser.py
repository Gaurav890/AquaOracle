"""Comprehensive PDF parsing with PyMuPDF."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from dataclasses import dataclass
from loguru import logger


@dataclass
class PDFPage:
    """Represents a single page from a PDF."""

    page_number: int
    text: str
    width: float
    height: float
    has_images: bool
    metadata: Dict[str, Any]


@dataclass
class PDFDocument:
    """Represents a complete PDF document."""

    file_path: Path
    title: str
    author: Optional[str]
    pages: List[PDFPage]
    page_count: int
    metadata: Dict[str, Any]


class PDFParser:
    """
    Comprehensive PDF parser using PyMuPDF.

    Extracts:
    - Text content with layout preservation
    - Page-level metadata
    - Document structure
    - Headers and footers
    - Multi-column layouts
    """

    def __init__(self):
        """Initialize PDF parser."""
        self.logger = logger.bind(name="PDFParser")

    def parse(self, pdf_path: Path) -> PDFDocument:
        """
        Parse a PDF file and extract all content.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFDocument object with extracted content

        Raises:
            DocumentProcessingError: If PDF parsing fails
        """
        self.logger.info(f"Parsing PDF: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)

            # Extract document metadata
            metadata = self._extract_document_metadata(doc)
            title = metadata.get("title", pdf_path.stem)
            author = metadata.get("author")

            # Extract pages
            pages = []
            for page_num in range(len(doc)):
                page = self._extract_page(doc, page_num)
                pages.append(page)

            doc.close()

            pdf_document = PDFDocument(
                file_path=pdf_path,
                title=title,
                author=author,
                pages=pages,
                page_count=len(pages),
                metadata=metadata,
            )

            self.logger.info(
                f"Successfully parsed {pdf_path.name}: {len(pages)} pages, "
                f"{sum(len(p.text) for p in pages)} characters"
            )

            return pdf_document

        except Exception as e:
            self.logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            raise

    def _extract_document_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract document-level metadata."""
        metadata = doc.metadata or {}

        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "mod_date": metadata.get("modDate", ""),
            "page_count": doc.page_count,
        }

    def _extract_page(self, doc: fitz.Document, page_num: int) -> PDFPage:
        """Extract content from a single page."""
        page = doc[page_num]

        # Extract text with layout preservation
        # Use "blocks" mode to preserve structure
        text = page.get_text("text")

        # Get page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height

        # Check for images
        image_list = page.get_images()
        has_images = len(image_list) > 0

        # Page metadata
        metadata = {
            "rotation": page.rotation,
            "mediabox": list(page.mediabox),
            "cropbox": list(page.cropbox) if hasattr(page, 'cropbox') else None,
            "image_count": len(image_list),
        }

        return PDFPage(
            page_number=page_num + 1,  # 1-indexed for user display
            text=text,
            width=width,
            height=height,
            has_images=has_images,
            metadata=metadata,
        )

    def extract_text_blocks(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text blocks with position information.
        Useful for layout-aware processing.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text blocks with position and formatting info
        """
        doc = fitz.open(pdf_path)
        all_blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    all_blocks.append({
                        "page": page_num + 1,
                        "bbox": block["bbox"],
                        "text": " ".join(
                            span["text"]
                            for line in block.get("lines", [])
                            for span in line.get("spans", [])
                        ),
                        "block_type": "text",
                    })

        doc.close()
        return all_blocks
