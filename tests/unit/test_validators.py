"""Tests for image validation."""

import io

import pytest
from PIL import Image

from image_identification.domain.errors import ImageValidationError
from image_identification.images.validators import validate_content_type, validate_image_bytes


class TestValidateContentType:
    """Test content-type validation."""

    def test_valid_png(self) -> None:
        validate_content_type("image/png", "test", "http://test")

    def test_valid_jpeg(self) -> None:
        validate_content_type("image/jpeg", "test", "http://test")

    def test_valid_webp(self) -> None:
        validate_content_type("image/webp", "test", "http://test")

    def test_valid_with_charset(self) -> None:
        validate_content_type("image/png; charset=utf-8", "test", "http://test")

    def test_invalid_html(self) -> None:
        with pytest.raises(ImageValidationError):
            validate_content_type("text/html", "test", "http://test")

    def test_none_content_type_skips(self) -> None:
        """None content type should skip validation."""
        validate_content_type(None, "test", "http://test")


class TestValidateImageBytes:
    """Test image bytes validation."""

    def test_valid_png_bytes(self) -> None:
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = validate_image_bytes(buf.getvalue(), "test", "http://test")
        assert result.size == (10, 10)

    def test_valid_jpeg_bytes(self) -> None:
        img = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = validate_image_bytes(buf.getvalue(), "test", "http://test")
        assert result.size == (10, 10)

    def test_invalid_bytes(self) -> None:
        with pytest.raises(ImageValidationError, match="Invalid image bytes"):
            validate_image_bytes(b"not an image", "test", "http://test")

    def test_empty_bytes(self) -> None:
        with pytest.raises(ImageValidationError):
            validate_image_bytes(b"", "test", "http://test")
