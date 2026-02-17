"""Image validation — ensures fetched bytes are safe and valid.

Treats all remote bytes as UNTRUSTED.
"""

from __future__ import annotations

import io

from PIL import Image

from product_image_id.domain.errors import ImageValidationError

# Allowed MIME types for product images
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)


def validate_content_type(content_type: str | None, item_code: str, url: str) -> None:
    """Validate that the HTTP content-type is an allowed image type.

    Raises ImageValidationError if not.
    """
    if content_type is None:
        return  # Skip validation if no content type header

    # Content-Type may include charset, e.g. "image/png; charset=utf-8"
    mime = content_type.split(";")[0].strip().lower()
    if mime not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            item_code=item_code,
            url=url,
            reason=f"Unexpected content type: {content_type}",
        )


def validate_image_bytes(image_bytes: bytes, item_code: str, url: str) -> Image.Image:
    """Validate that bytes can be opened as a valid image.

    Parameters
    ----------
    image_bytes:
        Raw bytes from the HTTP response.
    item_code:
        For error context.
    url:
        For error context.

    Returns
    -------
    PIL.Image.Image
        The opened (and verified) image.

    Raises
    ------
    ImageValidationError
        If the bytes cannot be parsed as a valid image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify it's a valid image structure
        # Re-open because verify() can leave the image in an unusable state
        img = Image.open(io.BytesIO(image_bytes))
        return img
    except Exception as exc:
        raise ImageValidationError(
            item_code=item_code,
            url=url,
            reason=f"Invalid image bytes: {exc}",
        ) from exc
