"""Tests for category classification (shoe / apparel / accessory).

Priority order: name keywords > family name.
"""

import pytest

from image_identification.domain.models import Category
from image_identification.features.category_classifier import classify_category


class TestClassifyCategory:
    """Test rule-based category classification."""

    # ------------------------------------------------------------------
    # Shoes (by family fallback — no apparel/accessory keywords in name)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "family",
        [
            "Cloud",
            "The Roger",
            "Monster",
            "Runner",
            "Trail",
            "Nova",
            "Eclipse",
            "Flyer",
            "Swift",
            "Ultra",
            "X",
            "Venture",
        ],
    )
    def test_shoe_families(self, family: str) -> None:
        """Known shoe families should classify as SHOE when name has no keywords."""
        result = classify_category(
            item_name="Some Product 1 M Black",
            family=family,
        )
        assert result == Category.SHOE

    # ------------------------------------------------------------------
    # Shoes (by name keywords — e.g. "Cloud 6 Waterproof")
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("item_name", "family"),
        [
            ("Cloud 5 M Black", "Cloud"),
            ("Cloud 6 Waterproof", "All Day"),  # family is wrong, name wins
            ("Cloudaway 1 M Ivory | Pearl", "Away"),
            ("Cloudmonster 2 W Eclipse | Magnet", "Monster"),
            ("Cloudboom Echo 3 M Undyed-White | Mint", "Boom"),
            ("THE ROGER Centre Court 1 W White | Gum", "The Roger"),
        ],
    )
    def test_shoe_by_name(self, item_name: str, family: str) -> None:
        result = classify_category(item_name=item_name, family=family)
        assert result == Category.SHOE

    # ------------------------------------------------------------------
    # Apparel (by name keywords — TAKES PRIORITY over family)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("item_name", "family"),
        [
            ("Active Jacket 1 M Undyed-White", "All Day"),
            ("Active Pants 1 M Black", "All Day"),
            ("Performance Tee 1 W Glacier", "Performance"),
            ("Weather Jacket OAC 1 W Black", "Performance"),
            ("Club Hoodie 1 W Crater", "Performance"),
            ("Active Shorts 1 M Black", "All Day"),
            ("Studio Tight 1 W Black", "Studio"),
            # Ascent family but apparel keywords in name
            ("Trek Pants 1 M Black", "Ascent"),
            ("Insulator Jacket 1 M Black", "Ascent"),
            ("Insulator Jacket 1 W Cocoa | Black", "Ascent"),
        ],
    )
    def test_apparel_items(self, item_name: str, family: str) -> None:
        result = classify_category(item_name=item_name, family=family)
        assert result == Category.APPAREL

    # ------------------------------------------------------------------
    # Accessories (by name keywords)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("item_name", "family"),
        [
            ("Merino Beanie 1 U Lunar", "All Day"),
            ("Moulded Cap 1 U Indigo", "All Day"),
            ("All-Day Sock 1 W Doe | Moss", "All Day"),
            ("Performance Mid Sock 1 W Rose | Flamingo", "Socks"),
            # Ascent family but accessory keywords in name
            ("Explorer Merino Sock 1 M Black | Glacier", "Ascent"),
            ("Explorer Merino Beanie 1 U Rock | Black", "Ascent"),
        ],
    )
    def test_accessory_items(self, item_name: str, family: str) -> None:
        result = classify_category(item_name=item_name, family=family)
        assert result == Category.ACCESSORY

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_unknown_product(self) -> None:
        """Completely unrecognized products should return UNKNOWN."""
        result = classify_category(
            item_name="Mystery Product 1 U Blue",
            family="Unknown Family",
        )
        assert result == Category.UNKNOWN

    def test_name_keywords_take_priority_over_family(self) -> None:
        """Name keywords should ALWAYS win over family classification.

        If an item has "jacket" in the name, it's apparel even if the
        family is a shoe family.
        """
        result = classify_category(
            item_name="Cloud Jacket Edition 1 M Black",
            family="Cloud",
        )
        assert result == Category.APPAREL

    def test_name_sock_in_shoe_family(self) -> None:
        """Sock keyword should win over shoe family."""
        result = classify_category(
            item_name="Cloud Running Sock 1 M Black",
            family="Cloud",
        )
        assert result == Category.ACCESSORY
