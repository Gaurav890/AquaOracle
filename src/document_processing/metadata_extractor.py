"""Extract metadata from documents."""

from pathlib import Path
from typing import Dict, Any, Optional
import re
from datetime import datetime
from loguru import logger


class MetadataExtractor:
    """Extract metadata from PDF documents."""

    def __init__(self):
        """Initialize metadata extractor."""
        self.logger = logger.bind(name="MetadataExtractor")

    def extract(self, pdf_metadata: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """
        Extract and enrich document metadata.

        Args:
            pdf_metadata: Raw PDF metadata dict
            file_path: Path to PDF file

        Returns:
            Enriched metadata dictionary
        """
        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "processed_at": datetime.now().isoformat(),
        }

        # Extract from PDF metadata
        metadata["title"] = self._clean_text(pdf_metadata.get("title", ""))
        metadata["author"] = self._clean_text(pdf_metadata.get("author", ""))
        metadata["subject"] = self._clean_text(pdf_metadata.get("subject", ""))
        metadata["keywords"] = self._clean_text(pdf_metadata.get("keywords", ""))
        metadata["creator"] = self._clean_text(pdf_metadata.get("creator", ""))
        metadata["page_count"] = pdf_metadata.get("page_count", 0)

        # Parse dates
        metadata["creation_date"] = self._parse_pdf_date(pdf_metadata.get("creation_date", ""))
        metadata["mod_date"] = self._parse_pdf_date(pdf_metadata.get("mod_date", ""))

        # If title is empty, use filename
        if not metadata["title"]:
            metadata["title"] = file_path.stem

        # Extract additional info from filename
        filename_info = self._parse_filename(file_path.name)
        metadata.update(filename_info)

        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def _parse_pdf_date(self, date_str: str) -> Optional[str]:
        """Parse PDF date format to ISO format."""
        if not date_str:
            return None

        # PDF date format: D:YYYYMMDDHHmmSSOHH'mm
        match = re.match(r'D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?', date_str)
        if match:
            year, month, day = match.groups()[:3]
            try:
                dt = datetime(int(year), int(month), int(day))
                return dt.date().isoformat()
            except:
                pass

        return None

    def _parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from filename.

        Examples:
        - "CDC, 2017.pdf" -> organization: CDC, year: 2017
        - "WHO, 2011.pdf" -> organization: WHO, year: 2011
        - "OSHA technical manual, 3, 7, 1999.pdf" -> organization: OSHA, year: 1999
        """
        metadata = {}

        # Remove extension
        name = Path(filename).stem

        # Try to extract year (4 digits)
        year_match = re.search(r'\b(19|20)\d{2}\b', name)
        if year_match:
            metadata["year"] = int(year_match.group())
        else:
            metadata["year"] = None

        # Try to extract organization (typically at start before comma)
        org_match = re.match(r'^([^,]+)', name)
        if org_match:
            org = org_match.group(1).strip()
            # Clean up common patterns
            org = re.sub(r'\s+\d+$', '', org)  # Remove trailing numbers
            metadata["organization"] = org
        else:
            metadata["organization"] = None

        return metadata

    def create_doc_id(self, file_path: Path) -> str:
        """
        Create unique document ID from file path.

        Args:
            file_path: Path to document

        Returns:
            Document ID
        """
        # Use filename without extension as ID
        # Replace spaces and special chars with underscore
        doc_id = file_path.stem
        doc_id = re.sub(r'[^\w\-]', '_', doc_id)
        doc_id = re.sub(r'_+', '_', doc_id)  # Remove duplicate underscores
        return doc_id
