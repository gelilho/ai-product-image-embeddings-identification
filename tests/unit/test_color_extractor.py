"""Tests for color extraction from product item names."""

import pytest

from product_image_id.features.color_extractor import extract_color, normalize_color


class TestExtractColor:
    """Test the regex-based color extraction from On Running product names."""

    @pytest.mark.parametrize(
        ("item_name", "expected_color"),
        [
            # Standard shoe names
            ("Cloud 5 M Black", "Black"),
            ("Cloud 5 W All Black", "All Black"),
            ("Cloudgo 1 W Black | Eclipse", "Black | Eclipse"),
            ("Cloudmonster 2 M Undyed-White | Mint", "Undyed-White | Mint"),
            # Apparel
            ("Active Jacket 1 M Undyed-White", "Undyed-White"),
            ("Active Pants 1 M Black", "Black"),
            # Accessories
            ("Merino Beanie 1 U Lunar", "Lunar"),
            ("Moulded Cap 1 U Indigo", "Indigo"),
            ("All-Day Sock 1 W Doe | Moss", "Doe | Moss"),
            ("All-Day Sock 1 M White | Wash", "White | Wash"),
            # Multi-word product names
            ("THE ROGER Centre Court 1 W White | Gum", "White | Gum"),
            ("Cloud X 3 W Ivory | Alloy", "Ivory | Alloy"),
            # Kids
            ("Cloud Play 1 K Glacier | White", "Glacier | White"),
        ],
    )
    def test_known_patterns(self, item_name: str, expected_color: str) -> None:
        assert extract_color(item_name) == expected_color

    def test_empty_name_returns_unknown(self) -> None:
        assert extract_color("") == "Unknown"

    def test_whitespace_name_returns_unknown(self) -> None:
        assert extract_color("   ") == "Unknown"

    def test_no_gender_indicator(self) -> None:
        """Names without a recognized gender indicator should fallback."""
        result = extract_color("Random Product Name")
        assert isinstance(result, str)


class TestNormalizeColor:
    """Test color normalization for comparison."""

    def test_lowercase(self) -> None:
        assert normalize_color("Black") == "black"

    def test_pipe_separator(self) -> None:
        assert normalize_color("Black | Eclipse") == "black|eclipse"

    def test_strips_whitespace(self) -> None:
        assert normalize_color("  Black  ") == "black"
