"""Canonical DOI normalization utilities.

Every module that needs to clean or compare DOIs should use these helpers
instead of re-implementing string replacement.
"""

from __future__ import annotations

# Ordered from longest to shortest so that e.g. "https://doi.org/"
# is stripped before "doi.org/".
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "dx.doi.org/",
    "doi:",
    "DOI:",
)


def normalize_doi(doi: str) -> str:
    """Strip common resolver prefixes and whitespace from a DOI string.

    Handles the most common prefix variants found in publisher metadata,
    citation exports, and web URLs.

    Examples:
        >>> normalize_doi("https://doi.org/10.1000/foo")
        '10.1000/foo'
        >>> normalize_doi("  10.1000/foo  ")
        '10.1000/foo'
        >>> normalize_doi("DOI:10.1000/foo")
        '10.1000/foo'
        >>> normalize_doi("http://dx.doi.org/10.1000/foo")
        '10.1000/foo'
    """
    if not doi:
        return ""
    cleaned = doi.strip()
    for prefix in _DOI_PREFIXES:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):]
            break  # Only strip the first matching prefix
    return cleaned.strip()


def normalize_doi_key(doi: object) -> str:
    """Normalize a DOI for case-insensitive comparison and dictionary keys."""
    return normalize_doi(str(doi or "")).lower()


def doi_from_filename_stem(stem: str) -> str:
    """Recover a DOI-like string from a filesystem-safe filename stem."""
    doi = str(stem or "").replace("_", "/")
    return normalize_doi(doi.removeprefix("doi.org/"))
