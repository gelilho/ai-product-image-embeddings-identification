"""Deterministic URL builder for product images.

Given an item_code, builds the full image URL using the pattern:
    {base_url}/{item_code}{suffix}

Example:
    item_code = "59.98842"
    → https://onretailimages.blob.core.windows.net/prod-mediaserver/Products/59.98842_000_001.png

Works for both legacy (NNN.NNNNN) and new (e.g. 1MD10060553) code formats.
"""

from __future__ import annotations

from image_identification.config import FetcherConfig


class UrlBuilder:
    """Builds product image URLs from item codes."""

    def __init__(self, config: FetcherConfig | None = None) -> None:
        cfg = config or FetcherConfig()
        self._base_url = cfg.base_url.rstrip("/")
        self._suffix = cfg.image_suffix

    def build(self, item_code: str) -> str:
        """Build the full image URL for the given item code.

        Parameters
        ----------
        item_code:
            The product item code (e.g. "59.98842" or "1MD10060553").

        Returns
        -------
        str
            Fully qualified image URL.

        Raises
        ------
        ValueError
            If item_code is empty or blank.
        """
        code = item_code.strip()
        if not code:
            raise ValueError("item_code must not be empty")
        return f"{self._base_url}/{code}{self._suffix}"

    def build_many(self, item_codes: list[str]) -> dict[str, str]:
        """Build URLs for a batch of item codes.

        Returns
        -------
        dict[str, str]
            Mapping of item_code → URL.
        """
        return {code: self.build(code) for code in item_codes}
