"""Category classification: shoe vs apparel vs accessory.

Uses a rule-based approach combining item name keywords and family name.
Item name keywords take PRIORITY over family — e.g. "Trek Pants" in the
"Ascent" family is apparel, not a shoe.
"""

from __future__ import annotations

from image_identification.domain.models import Category

# ---------------------------------------------------------------------------
# Keyword sets — derived from golden dataset analysis
# ---------------------------------------------------------------------------

# Keywords in item names that indicate apparel (HIGHEST priority)
APPAREL_KEYWORDS: frozenset[str] = frozenset(
    {
        "jacket",
        "shirt",
        "tee",
        "shorts",
        "pants",
        "tight",
        "tights",
        "dress",
        "skirt",
        "vest",
        "hoodie",
        "sweater",
        "top",
        "bra",
        "legging",
        "leggings",
        "tank",
        "crop",
        "polo",
        "parka",
        "anorak",
        "gilet",
        "sweatshirt",
        "insulator",
    }
)

# Keywords in item names that indicate accessories
ACCESSORY_KEYWORDS: frozenset[str] = frozenset(
    {
        "beanie",
        "cap",
        "sock",
        "socks",
        "hat",
        "bag",
        "belt",
        "glove",
        "gloves",
        "scarf",
        "headband",
        "visor",
        "backpack",
        "duffle",
        "towel",
    }
)

# Keywords in item names that indicate shoes
SHOE_NAME_KEYWORDS: frozenset[str] = frozenset(
    {
        "cloud",
        "cloudaway",
        "cloudmonster",
        "cloudnova",
        "cloudrunner",
        "cloudswift",
        "cloudsurfer",
        "cloudflow",
        "cloudstratus",
        "cloudultra",
        "cloudventure",
        "cloudrock",
        "cloudeclipse",
        "cloudflyer",
        "cloudboom",
        "cloudgo",
        "cloudvista",
        "cloudwander",
        "cloudrift",
        "cloudhero",
        "roger",
    }
)

# Families that are ALWAYS shoes (used as fallback AFTER name keyword check)
SHOE_FAMILIES: frozenset[str] = frozenset(
    {
        "Away",
        "Boom",
        "Cloud",
        "Club",
        "Courtside",
        "Eclipse",
        "Flow",
        "Flyer",
        "Focus",
        "Go",
        "Hero",
        "Monster",
        "Nova",
        "Rift",
        "Runner",
        "Stratus",
        "Surfer",
        "Swift",
        "The Roger",
        "Trail",
        "Train",
        "Ultra",
        "Venture",
        "Vista",
        "Wander",
        "X",
    }
)


def classify_category(
    item_name: str,
    family: str,
    vertical: str = "",
) -> Category:
    """Classify a product as shoe, apparel, or accessory.

    Priority order:
      1. Item name → apparel keywords (jacket, pants, shirt, etc.)
      2. Item name → accessory keywords (beanie, cap, sock, etc.)
      3. Item name → shoe keywords (cloud*, roger, etc.)
      4. Family name → shoe families (Away, Boom, Cloud, etc.)
      5. Unknown

    This ordering ensures that "Insulator Jacket" in the "Ascent" family
    is correctly classified as apparel, not shoe.
    """
    name_lower = item_name.strip().lower()
    family_clean = family.strip()

    # Tokenize the item name for word-level matching
    name_words = set(name_lower.replace("-", " ").split())

    # Rule 1: Apparel keywords in item name (HIGHEST priority)
    if name_words & APPAREL_KEYWORDS:
        return Category.APPAREL

    # Rule 2: Accessory keywords in item name
    if name_words & ACCESSORY_KEYWORDS:
        return Category.ACCESSORY

    # Rule 3: Shoe keywords in item name (e.g. "Cloud 6 Waterproof")
    # Check both exact word match and prefix match (for compound names)
    if name_words & SHOE_NAME_KEYWORDS:
        return Category.SHOE
    # Also check if any word starts with a shoe keyword (e.g. "cloudaway")
    for word in name_words:
        for kw in SHOE_NAME_KEYWORDS:
            if word.startswith(kw):
                return Category.SHOE

    # Rule 4: Family → shoe mapping (fallback)
    if family_clean in SHOE_FAMILIES:
        return Category.SHOE

    # Rule 5: Socks family
    if family_clean.lower() == "socks" or "sock" in name_lower:
        return Category.ACCESSORY

    return Category.UNKNOWN
