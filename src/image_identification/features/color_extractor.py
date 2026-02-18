"""Color extraction from product item names.

On Running naming convention:
    "{Product} {Version} {GenderIndicator} {Color}"

Examples:
    "Cloud 5 M Black"              → "Black"
    "Cloudgo 1 W Black | Eclipse"  → "Black | Eclipse"
    "All-Day Sock 1 W Doe | Moss"  → "Doe | Moss"
    "Merino Beanie 1 U Lunar"      → "Lunar"
    "Active Jacket 1 M Undyed-White" → "Undyed-White"

Gender indicators: M (Men), W (Women), U (Unisex), K/Jr (Kids)
"""

from __future__ import annotations

import re

# Gender indicators that appear in item names
_GENDER_INDICATORS: frozenset[str] = frozenset({"M", "W", "U", "K", "Jr"})

# Regex: match everything after the last gender indicator + space
# Pattern: <anything> <version_number> <gender_indicator> <color...>
_NAME_PATTERN = re.compile(r"^.+?\s+\d+\s+(?:M|W|U|K|Jr)\s+(.+)$")


def extract_color(item_name: str) -> str:
    """Extract the color portion from a product item name.

    Parameters
    ----------
    item_name:
        The full product name, e.g. "Cloud 5 M Black | Eclipse".

    Returns
    -------
    str
        The extracted color string, or "Unknown" if parsing fails.
    """
    name = item_name.strip()
    if not name:
        return "Unknown"

    match = _NAME_PATTERN.match(name)
    if match:
        return match.group(1).strip()

    # Fallback: try splitting on known gender indicators from the right
    parts = name.split()
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in _GENDER_INDICATORS and i < len(parts) - 1:
            return " ".join(parts[i + 1 :]).strip()

    return "Unknown"


def normalize_color(color: str) -> str:
    """Normalize a color string for comparison.

    Lowercases, strips whitespace, and normalizes separators.
    """
    return color.strip().lower().replace(" | ", "|").replace("  ", " ")
