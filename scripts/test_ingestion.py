#!/usr/bin/env python3
"""Test document ingestion on a single PDF."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging_config import setup_logging
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import Chunker
from src.document_processing.metadata_extractor import MetadataExtractor

setup_logging(log_level="INFO")


def test_parse_pdf(pdf_path: Path):
    """Test parsing a single PDF."""
    print(f"\n{'='*60}")
    print(f"Testing PDF Parsing: {pdf_path.name}")
    print(f"{'='*60}\n")

    # Parse PDF
    parser = PDFParser()
    doc = parser.parse(pdf_path)

    print("✓ Document parsed successfully")
    print(f"  - Title: {doc.title}")
    print(f"  - Pages: {doc.page_count}")
    print(f"  - Total characters: {sum(len(p.text) for p in doc.pages):,}")

    # Extract metadata
    metadata_extractor = MetadataExtractor()
    metadata = metadata_extractor.extract(doc.metadata, pdf_path)
    doc_id = metadata_extractor.create_doc_id(pdf_path)

    print("\n✓ Metadata extracted")
    print(f"  - Document ID: {doc_id}")
    print(f"  - Organization: {metadata.get('organization', 'N/A')}")
    print(f"  - Year: {metadata.get('year', 'N/A')}")

    # Chunk document
    chunker = Chunker(max_chunk_size=1024, overlap=0.2)
    pages = [(page.page_number, page.text) for page in doc.pages]
    chunks = chunker.chunk_document(
        pages=pages,
        doc_id=doc_id,
        metadata=metadata,
    )

    print("\n✓ Document chunked")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Avg tokens per chunk: {sum(c.token_count for c in chunks) / len(chunks):.0f}")

    # Show first chunk
    print(f"\n{'='*60}")
    print("First Chunk Preview:")
    print(f"{'='*60}")
    print(chunks[0].text[:500] + "...")

    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_ingestion.py <path-to-pdf>")
        print("\nExample:")
        print('  python scripts/test_ingestion.py "soc/CDC, 2017.pdf"')
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: Not a PDF file: {pdf_path}")
        sys.exit(1)

    test_parse_pdf(pdf_path)
