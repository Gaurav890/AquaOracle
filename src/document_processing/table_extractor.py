"""Table extraction from PDFs using pdfplumber."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
from dataclasses import dataclass
from loguru import logger


@dataclass
class Table:
    """Represents an extracted table from PDF."""

    page_number: int
    bbox: tuple  # (x0, y0, x1, y1)
    data: List[List[str]]  # 2D array of cell values
    row_count: int
    col_count: int
    has_header: bool
    caption: Optional[str] = None


class TableExtractor:
    """Extract tables from PDFs using pdfplumber."""

    def __init__(self,
                 table_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize table extractor.

        Args:
            table_settings: Custom settings for table detection
        """
        self.logger = logger.bind(name="TableExtractor")
        self.table_settings = table_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 3,
        }

    def extract_tables(self, pdf_path: Path) -> List[Table]:
        """
        Extract all tables from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted tables
        """
        self.logger.info(f"Extracting tables from: {pdf_path}")

        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = self._extract_page_tables(page, page_num)
                    tables.extend(page_tables)

            self.logger.info(f"Extracted {len(tables)} tables from {pdf_path.name}")
            return tables

        except Exception as e:
            self.logger.error(f"Failed to extract tables from {pdf_path}: {e}")
            return []

    def _extract_page_tables(self, page, page_num: int) -> List[Table]:
        """Extract tables from a single page."""
        page_tables = []

        try:
            # Find tables using pdfplumber's table detection
            tables = page.find_tables(self.table_settings)

            for table in tables:
                # Extract table data
                table_data = table.extract()

                if table_data and len(table_data) > 0:
                    # Clean empty rows
                    table_data = [row for row in table_data if any(cell for cell in row)]

                    if len(table_data) > 0:
                        # Determine if first row is header
                        has_header = self._is_header_row(table_data[0])

                        page_tables.append(Table(
                            page_number=page_num,
                            bbox=table.bbox,
                            data=table_data,
                            row_count=len(table_data),
                            col_count=len(table_data[0]) if table_data else 0,
                            has_header=has_header,
                        ))

        except Exception as e:
            self.logger.warning(f"Error extracting tables from page {page_num}: {e}")

        return page_tables

    def _is_header_row(self, row: List[str]) -> bool:
        """Heuristic to determine if a row is a header."""
        if not row:
            return False

        # Header detection heuristics:
        # 1. All cells are non-empty
        # 2. Cells are relatively short (< 50 chars)
        # 3. No numeric-only cells

        non_empty = sum(1 for cell in row if cell and cell.strip())
        if non_empty == 0:
            return False

        short_cells = sum(1 for cell in row if cell and len(cell) < 50)
        numeric_cells = sum(1 for cell in row if cell and cell.strip().replace('.', '').replace(',', '').isdigit())

        # Likely a header if most cells are short and not purely numeric
        return (short_cells / len(row) > 0.7) and (numeric_cells / len(row) < 0.5)

    def table_to_text(self, table: Table) -> str:
        """
        Convert table to readable text format.

        Args:
            table: Table object

        Returns:
            Formatted text representation
        """
        if not table.data:
            return ""

        lines = []
        lines.append(f"[Table on page {table.page_number}]")

        # Format as markdown-style table
        for i, row in enumerate(table.data):
            row_text = " | ".join(str(cell) if cell else "" for cell in row)
            lines.append(row_text)

            # Add separator after header
            if i == 0 and table.has_header:
                separator = " | ".join("---" for _ in row)
                lines.append(separator)

        return "\n".join(lines)

    def table_to_dict(self, table: Table) -> Dict[str, List]:
        """
        Convert table to dictionary format with headers as keys.

        Args:
            table: Table object

        Returns:
            Dictionary with column names as keys
        """
        if not table.data or len(table.data) < 2:
            return {}

        if table.has_header:
            headers = [str(h) if h else f"Column{i}" for i, h in enumerate(table.data[0])]
            data_rows = table.data[1:]
        else:
            headers = [f"Column{i}" for i in range(table.col_count)]
            data_rows = table.data

        result = {header: [] for header in headers}

        for row in data_rows:
            for i, cell in enumerate(row):
                if i < len(headers):
                    result[headers[i]].append(cell if cell else "")

        return result
