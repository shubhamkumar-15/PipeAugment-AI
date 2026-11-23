import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Set

from fastapi import HTTPException, UploadFile
from PIL import Image

SUPPORTED_FORMATS: Set[str] = {"JPEG", "PNG", "WEBP"}

FORMAT_EXTENSIONS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
}

CANONICAL_EXTENSIONS = {
    "JPEG": "jpg",
    "PNG": "png",
    "WEBP": "webp",
}

CONTENT_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True)
class ValidatedImage:
    raw_bytes: bytes
    detected_format: str
    canonical_ext: str
    content_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str
    sanitized_filename: str


def sanitize_original_filename(filename: str | None, canonical_ext: str) -> str:
    """Sanitize the original user-provided filename for metadata storage and display."""
    if not filename:
        return f"upload.{canonical_ext}"

    # Extract base filename to strip any directory traversal paths
    base_name = Path(filename).name

    # Remove null bytes, control characters, and unsafe filesystem characters
    cleaned = "".join(c for c in base_name if c.isprintable() and c not in '<>:"/\\|?*\0')
    cleaned = cleaned.strip(". ")

    if not cleaned:
        return f"upload.{canonical_ext}"

    return cleaned


async def validate_image_upload(
    file: UploadFile,
    max_upload_bytes: int,
    max_image_pixels: int,
) -> ValidatedImage:
    """
    Validate an uploaded image file according to PipeAugment AI requirements:
    1. Read in bounded chunks up to max_upload_bytes (reject with 413 if exceeded).
    2. Reject empty uploads (422).
    3. Verify actual image content with Pillow (do not trust extension or Content-Type).
    4. Check image integrity and pixel decoding.
    5. Treat Pillow decompression-bomb warnings/errors as rejection (422).
    6. Enforce pixel limit (422).
    7. Reject animated or multi-frame images (422).
    8. Reject unsupported actual image formats (415).
    9. Reject mismatches between supported filename extension and detected format (415).
    """
    chunk_size = 64 * 1024  # 64 KB chunks
    accumulated = bytearray()
    total_bytes = 0

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed upload size of {max_upload_bytes} bytes.",
                )
            accumulated.extend(chunk)
    finally:
        await file.close()

    if total_bytes == 0:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty.",
        )

    raw_bytes = bytes(accumulated)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    stream = io.BytesIO(raw_bytes)

    # Configure Pillow pixel limit
    Image.MAX_IMAGE_PIXELS = max_image_pixels

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            with Image.open(stream) as img:
                # Check for Pillow decompression bomb warnings
                for w in caught_warnings:
                    if issubclass(w.category, Image.DecompressionBombWarning):
                        raise HTTPException(
                            status_code=422,
                            detail="Image exceeds maximum decompression bomb limit.",
                        )

                detected_format = img.format
                if not detected_format or detected_format not in SUPPORTED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            f"Unsupported image format: '{detected_format or 'UNKNOWN'}'. "
                            "Supported formats: JPEG, PNG, WEBP."
                        ),
                    )

                width, height = img.size
                total_pixels = width * height
                if total_pixels > max_image_pixels:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Image pixel count ({total_pixels}) exceeds "
                            f"maximum allowed ({max_image_pixels})."
                        ),
                    )

                # Reject animated or multi-frame images
                is_animated = getattr(img, "is_animated", False)
                n_frames = getattr(img, "n_frames", 1)
                if is_animated or n_frames > 1:
                    raise HTTPException(
                        status_code=422,
                        detail="Animated or multi-frame images are not supported.",
                    )

                # Verify integrity
                img.verify()

            # Decoding pixel check: verify() invalidates the image object in Pillow,
            # so re-open from the beginning of the stream to decode pixels with load()
            stream.seek(0)
            with Image.open(stream) as img_decode:
                img_decode.load()

        except Image.DecompressionBombError:
            raise HTTPException(
                status_code=422,
                detail="Image exceeds maximum decompression bomb limit.",
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="Corrupt, invalid, or unreadable image content.",
            ) from exc

    # Extension validation against detected format
    raw_filename = file.filename or ""
    raw_ext = Path(raw_filename).suffix.lower()
    allowed_exts = FORMAT_EXTENSIONS[detected_format]

    if raw_ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Filename extension '{raw_ext}' does not match detected format '{detected_format}'. "
                f"Expected one of: {sorted(allowed_exts)}."
            ),
        )

    canonical_ext = CANONICAL_EXTENSIONS[detected_format]
    content_type = CONTENT_TYPES[detected_format]
    sanitized_filename = sanitize_original_filename(raw_filename, canonical_ext)

    return ValidatedImage(
        raw_bytes=raw_bytes,
        detected_format=detected_format,
        canonical_ext=canonical_ext,
        content_type=content_type,
        width=width,
        height=height,
        size_bytes=total_bytes,
        sha256=sha256,
        sanitized_filename=sanitized_filename,
    )
