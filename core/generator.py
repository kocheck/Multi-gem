"""Gemini API interaction — image generation with reference image support."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from core.config import AppConfig
from core.csv_parser import PromptRow, get_effective_reference_images
from core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Pixel dimensions for each resolution tier (used for informational purposes;
# actual enforcement is done by the model itself)
RESOLUTION_PIXELS: dict[str, int] = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}


class GenerationResult:
    """Holds the outcome of a single image generation request."""

    def __init__(
        self,
        row: PromptRow,
        image_data: Optional[bytes] = None,
        mime_type: str = "image/png",
        error: Optional[str] = None,
    ) -> None:
        self.row = row
        self.image_data = image_data
        self.mime_type = mime_type
        self.error = error

    @property
    def success(self) -> bool:
        return self.image_data is not None and self.error is None


class ImageGenerator:
    """
    Wraps the Google Gemini API for image generation.

    Handles:
    - Reference image loading (per-prompt + group, merged, up to 14 images)
    - Prompt construction (with style instructions and aspect ratio hints)
    - Rate limiting and exponential backoff via RateLimiter
    - Per-request error isolation (never raises, returns GenerationResult with error)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.rate_limiter = RateLimiter(
            delay_between_requests=config.delay_between_requests,
            max_retries=config.max_retries,
        )

    def generate(self, row: PromptRow) -> GenerationResult:
        """
        Generate an image for a single PromptRow.

        This method never raises — all exceptions are caught and returned as
        a GenerationResult with error set.

        Args:
            row: Validated PromptRow containing prompt, model, aspect ratio, etc.

        Returns:
            GenerationResult with image_data on success, error message on failure.
        """
        try:
            return self._generate_with_retry(row)
        except Exception as exc:
            logger.error("Unhandled error generating row %d: %s", row.row_index, exc)
            return GenerationResult(row=row, error=str(exc))

    def _generate_with_retry(self, row: PromptRow) -> GenerationResult:
        """Internal: build request and call API with rate limiting."""
        prompt_text = _build_prompt(row)
        ref_paths = get_effective_reference_images(row, self.config)
        contents = _build_contents(prompt_text, ref_paths)

        generation_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            response_mime_type="text/plain",
        )

        model_id = row.model

        def _call() -> GenerationResult:
            logger.debug(
                "Calling model=%s prompt=%r aspect_ratio=%s",
                model_id,
                prompt_text[:80],
                row.aspect_ratio,
            )
            response = self.client.models.generate_content(
                model=model_id,
                contents=contents,
                config=generation_config,
            )

            # Extract image bytes from response parts
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    return GenerationResult(
                        row=row,
                        image_data=part.inline_data.data,
                        mime_type=part.inline_data.mime_type,
                    )

            # No image returned — surface any text response for diagnosis
            text_parts = [
                p.text for p in response.candidates[0].content.parts if p.text
            ]
            reason = "; ".join(text_parts) if text_parts else "No image in response."
            return GenerationResult(row=row, error=f"No image returned: {reason}")

        return self.rate_limiter.call_with_retry(_call)


# ---------------------------------------------------------------------------
# Prompt building helpers
# ---------------------------------------------------------------------------


def _build_prompt(row: PromptRow) -> str:
    """Construct the full prompt string from a PromptRow."""
    parts = [row.prompt.strip()]

    if row.style_instructions:
        parts.append(row.style_instructions.strip())

    # Embed aspect ratio in the prompt as a soft hint for models that support it
    parts.append(f"Aspect ratio: {row.aspect_ratio}.")

    if row.negative_prompt:
        parts.append(f"Avoid: {row.negative_prompt.strip()}.")

    return " ".join(parts)


def _build_contents(
    prompt_text: str, ref_image_paths: list[str]
) -> list[types.Content]:
    """
    Build the contents list for the API call.

    If reference images are provided they are embedded as inline base64 parts
    preceding the text prompt, which is the standard technique for image-to-image
    and style-reference workflows with Gemini.

    Args:
        prompt_text: Full prompt string.
        ref_image_paths: List of local file paths to reference images.

    Returns:
        List of Content objects ready for generate_content().
    """
    parts: list[types.Part] = []

    for path_str in ref_image_paths:
        path = Path(path_str)
        mime_type = _detect_mime_type(path)
        image_bytes = path.read_bytes()
        parts.append(
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        )

    parts.append(types.Part.from_text(text=prompt_text))

    return [types.Content(role="user", parts=parts)]


def _detect_mime_type(path: Path) -> str:
    """Detect MIME type from file extension, defaulting to image/jpeg."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    # Fall back based on common extensions
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")
