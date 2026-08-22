"""Shared formatting for document/page/line citations.

Used by context assembly, prompt building, and response formatting so the
three don't each reimplement (and risk drifting on) the same logic.
"""

from typing import Any, Dict, List, Optional, Tuple


def format_page_citation(
    page_numbers: List[int],
    line_ranges: Optional[Dict[Any, Tuple[int, int]]] = None,
) -> str:
    """
    Format a chunk's page(s) and, when available, line range(s) for display.

    Args:
        page_numbers: Pages the chunk's text came from, in order.
        line_ranges: Optional page -> (first_line, last_line) map. Keys may
            be int or str since this sometimes round-trips through JSON.

    Returns:
        e.g. "Page 33, Line 12", "Page 33, Lines 12-18",
        "Page 33, Lines 40-52; Page 34, Lines 1-8", or "Pages 33-34" if no
        line range is available for a page.
    """
    if not page_numbers:
        return "Page Unknown"

    if not line_ranges:
        if len(page_numbers) == 1:
            return f"Page {page_numbers[0]}"
        return f"Pages {page_numbers[0]}-{page_numbers[-1]}"

    parts = []
    for page in page_numbers:
        line_range = line_ranges.get(page, line_ranges.get(str(page)))
        if not line_range:
            parts.append(f"Page {page}")
            continue
        start, end = line_range
        if start == end:
            parts.append(f"Page {page}, Line {start}")
        else:
            parts.append(f"Page {page}, Lines {start}-{end}")

    return "; ".join(parts)
