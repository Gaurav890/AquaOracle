"""Tests for document metadata and doc_id extraction."""

from pathlib import Path

from src.document_processing.metadata_extractor import MetadataExtractor


def test_create_doc_id_sanitizes_special_characters():
    extractor = MetadataExtractor()

    doc_id = extractor.create_doc_id(Path("CDC, 2017.pdf"))

    assert doc_id == "CDC_2017"


def test_create_doc_id_collapses_repeated_underscores():
    extractor = MetadataExtractor()

    doc_id = extractor.create_doc_id(Path("OSHA technical manual, 3, 7, 1999.pdf"))

    assert "__" not in doc_id


def test_parse_filename_extracts_organization_and_year():
    extractor = MetadataExtractor()

    info = extractor._parse_filename("CDC, 2017.pdf")

    assert info["organization"] == "CDC"
    assert info["year"] == 2017


def test_parse_filename_without_year_sets_none():
    extractor = MetadataExtractor()

    info = extractor._parse_filename("untitled-report.pdf")

    assert info["year"] is None


def test_extract_falls_back_to_filename_when_title_missing(tmp_path):
    extractor = MetadataExtractor()
    pdf_path = tmp_path / "WHO, 2011.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    metadata = extractor.extract({"title": "", "page_count": 5}, pdf_path)

    assert metadata["title"] == "WHO, 2011"
    assert metadata["organization"] == "WHO"
    assert metadata["year"] == 2011


def test_parse_pdf_date_valid():
    extractor = MetadataExtractor()

    assert extractor._parse_pdf_date("D:20170605120000") == "2017-06-05"


def test_parse_pdf_date_invalid_returns_none():
    extractor = MetadataExtractor()

    assert extractor._parse_pdf_date("not-a-date") is None
    assert extractor._parse_pdf_date("") is None
