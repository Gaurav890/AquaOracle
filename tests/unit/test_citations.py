"""Tests for the shared page/line citation formatter."""

from src.utils.citations import format_page_citation


def test_single_page_no_lines():
    assert format_page_citation([33]) == "Page 33"


def test_multi_page_no_lines():
    assert format_page_citation([33, 34]) == "Pages 33-34"


def test_no_pages():
    assert format_page_citation([]) == "Page Unknown"


def test_single_page_with_line_range():
    assert format_page_citation([33], {33: (12, 18)}) == "Page 33, Lines 12-18"


def test_single_page_with_single_line():
    assert format_page_citation([33], {33: (12, 12)}) == "Page 33, Line 12"


def test_multi_page_with_line_ranges():
    result = format_page_citation([33, 34], {33: (40, 52), 34: (1, 8)})
    assert result == "Page 33, Lines 40-52; Page 34, Lines 1-8"


def test_line_ranges_with_string_keys_from_json_roundtrip():
    # json.dumps/loads turns int dict keys into strings.
    assert format_page_citation([33], {"33": (12, 18)}) == "Page 33, Lines 12-18"


def test_page_missing_from_line_ranges_falls_back_to_page_only():
    result = format_page_citation([33, 34], {33: (12, 18)})
    assert result == "Page 33, Lines 12-18; Page 34"
