"""Output handling: file saving, manifest generation, thumbnail creation, HTML gallery."""

from __future__ import annotations

import csv
import datetime
import html
import io
import json
import logging
from pathlib import Path
from typing import Optional

from core.config import AppConfig
from core.csv_parser import PromptRow

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.csv"
GALLERY_FILENAME = "gallery.html"
IMAGES_SUBDIR = "images"
THUMBNAILS_SUBDIR = "thumbnails"

MANIFEST_FIELDNAMES = [
    "row_index",
    "prompt",
    "output_filename",
    "status",
    "error",
    "model",
    "aspect_ratio",
    "resolution",
    "group",
    "timestamp",
]


class OutputManager:
    """
    Manages the output directory structure for a batch run.

    Creates:
        <output_directory>/
        └── <timestamp_subfolder>/
            ├── manifest.csv
            ├── gallery.html
            ├── images/
            └── thumbnails/  (if generate_thumbnails is True)
    """

    def __init__(self, config: AppConfig, batch_timestamp: Optional[str] = None) -> None:
        self.config = config
        ts = batch_timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = Path(config.output_directory) / ts
        self.images_dir = self.run_dir / IMAGES_SUBDIR
        self.thumbnails_dir = self.run_dir / THUMBNAILS_SUBDIR
        self._manifest_rows: list[dict] = []
        self._manifest_writer: Optional[csv.DictWriter] = None
        self._manifest_file: Optional[io.TextIOWrapper] = None

    def setup(self) -> None:
        """Create all necessary output directories."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.config.generate_thumbnails:
            self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self._open_manifest()
        logger.info("Output directory: %s", self.run_dir)

    def _open_manifest(self) -> None:
        manifest_path = self.run_dir / MANIFEST_FILENAME
        self._manifest_file = manifest_path.open("w", newline="", encoding="utf-8")
        self._manifest_writer = csv.DictWriter(
            self._manifest_file, fieldnames=MANIFEST_FIELDNAMES
        )
        self._manifest_writer.writeheader()

    def close(self) -> None:
        """Flush and close the manifest file."""
        if self._manifest_file:
            self._manifest_file.flush()
            self._manifest_file.close()
            self._manifest_file = None

    def save_image(
        self,
        row: PromptRow,
        image_data: bytes,
        mime_type: str = "image/png",
    ) -> str:
        """
        Save image bytes to the images directory.

        Args:
            row: The PromptRow that generated this image.
            image_data: Raw image bytes.
            mime_type: MIME type of the image (used to determine extension).

        Returns:
            The filename (not full path) that was saved.
        """
        ext = _mime_to_ext(mime_type, self.config.output_format)
        filename = _resolve_filename(row, ext, self.images_dir)
        out_path = self.images_dir / filename
        out_path.write_bytes(image_data)
        logger.debug("Saved image: %s", out_path)

        if self.config.generate_thumbnails:
            self._save_thumbnail(image_data, filename, mime_type)

        return filename

    def _save_thumbnail(self, image_data: bytes, filename: str, mime_type: str) -> None:
        """Generate and save a thumbnail for the given image."""
        try:
            from PIL import Image  # type: ignore

            stem = Path(filename).stem
            ext = Path(filename).suffix
            thumb_filename = f"{stem}_thumb{ext}"
            thumb_path = self.thumbnails_dir / thumb_filename

            with Image.open(io.BytesIO(image_data)) as img:
                img.thumbnail((self.config.thumbnail_size, self.config.thumbnail_size))
                img.save(thumb_path)

            logger.debug("Saved thumbnail: %s", thumb_path)
        except ImportError:
            logger.warning("Pillow not installed — thumbnails skipped.")
        except Exception as exc:
            logger.warning("Failed to generate thumbnail for %s: %s", filename, exc)

    def record_success(self, row: PromptRow, filename: str) -> None:
        """Write a success entry to the manifest."""
        self._write_manifest(
            row=row,
            filename=filename,
            status="success",
            error="",
        )

    def record_error(self, row: PromptRow, error: str) -> None:
        """Write an error entry to the manifest."""
        self._write_manifest(
            row=row,
            filename="",
            status="error",
            error=error,
        )

    def _write_manifest(
        self, row: PromptRow, filename: str, status: str, error: str
    ) -> None:
        entry = {
            "row_index": row.row_index,
            "prompt": row.prompt,
            "output_filename": filename,
            "status": status,
            "error": error,
            "model": row.model,
            "aspect_ratio": row.aspect_ratio,
            "resolution": row.resolution,
            "group": row.group or "",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self._manifest_rows.append(entry)
        if self._manifest_writer:
            self._manifest_writer.writerow(entry)
            self._manifest_file.flush()  # type: ignore[union-attr]

    def generate_gallery(self) -> None:
        """Write an HTML gallery page listing all generated images with their prompts."""
        image_files = sorted(self.images_dir.glob("*.*"))
        thumb_rel = f"{THUMBNAILS_SUBDIR}/"
        img_rel = f"{IMAGES_SUBDIR}/"

        success_rows = [r for r in self._manifest_rows if r["status"] == "success"]
        error_rows = [r for r in self._manifest_rows if r["status"] == "error"]

        cards_html = ""
        for entry in success_rows:
            fname = entry["output_filename"]
            stem = Path(fname).stem
            thumb_name = f"{stem}_thumb{Path(fname).suffix}"
            thumb_src = f"{thumb_rel}{thumb_name}" if self.config.generate_thumbnails else f"{img_rel}{fname}"
            prompt_esc = html.escape(entry["prompt"])
            cards_html += f"""
      <div class="card">
        <a href="{img_rel}{fname}" target="_blank">
          <img src="{thumb_src}" alt="{prompt_esc}" loading="lazy" onerror="this.src='{img_rel}{fname}'">
        </a>
        <div class="caption">
          <p class="prompt">{prompt_esc}</p>
          <p class="meta">{html.escape(entry['model'])} · {html.escape(entry['aspect_ratio'])} · {html.escape(entry['resolution'])}</p>
        </div>
      </div>"""

        errors_html = ""
        if error_rows:
            errors_html = "<h2>Errors</h2><table><tr><th>Row</th><th>Prompt</th><th>Error</th></tr>"
            for entry in error_rows:
                errors_html += (
                    f"<tr><td>{html.escape(str(entry['row_index']))}</td>"
                    f"<td>{html.escape(entry['prompt'][:80])}</td>"
                    f"<td>{html.escape(entry['error'])}</td></tr>"
                )
            errors_html += "</table>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gemini Batch Output Gallery</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 1rem; }}
    h1 {{ text-align: center; color: #a78bfa; }}
    h2 {{ color: #f87171; }}
    .stats {{ text-align: center; color: #94a3b8; margin-bottom: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }}
    .card {{ background: #16213e; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px #0004; }}
    .card img {{ width: 100%; height: 200px; object-fit: cover; display: block; }}
    .caption {{ padding: .6rem; }}
    .prompt {{ font-size: .85rem; margin: 0 0 .3rem; }}
    .meta {{ font-size: .75rem; color: #94a3b8; margin: 0; }}
    table {{ border-collapse: collapse; width: 100%; background: #16213e; border-radius: 8px; }}
    th, td {{ padding: .5rem .75rem; text-align: left; border-bottom: 1px solid #334; font-size: .85rem; }}
    th {{ color: #a78bfa; }}
  </style>
</head>
<body>
  <h1>🎨 Gemini Batch Gallery</h1>
  <p class="stats">{len(success_rows)} succeeded · {len(error_rows)} failed</p>
  <div class="grid">{cards_html}
  </div>
  {errors_html}
</body>
</html>
"""
        gallery_path = self.run_dir / GALLERY_FILENAME
        gallery_path.write_text(html_content, encoding="utf-8")
        logger.info("Gallery written: %s", gallery_path)

    def load_completed_filenames(self) -> set[str]:
        """
        Load successfully completed output filenames from an existing manifest.

        Used by --resume to skip already-completed prompts.

        Returns:
            Set of output_filename values with status == 'success'.
        """
        manifest_path = self.run_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            return set()

        completed: set[str] = set()
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "success" and row.get("output_filename"):
                    completed.add(row["output_filename"])
        return completed

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_FILENAME

    @property
    def gallery_path(self) -> Path:
        return self.run_dir / GALLERY_FILENAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mime_to_ext(mime_type: str, preferred_format: str) -> str:
    """Return file extension for a MIME type, falling back to preferred_format."""
    mime_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    ext = mime_map.get(mime_type, preferred_format)
    return ext


def _resolve_filename(row: PromptRow, ext: str, images_dir: Path) -> str:
    """
    Determine the output filename for an image.

    Uses the row's output_filename if provided, otherwise auto-generates from
    the prompt (slugified). Appends a counter suffix if the file already exists.

    Args:
        row: The PromptRow.
        ext: File extension without leading dot.
        images_dir: Directory where images are saved (used to check uniqueness).

    Returns:
        Filename string (e.g., 'my_image.png').
    """
    if row.output_filename:
        base = _slugify(row.output_filename)
    else:
        # Auto-generate from prompt (first 50 chars, slugified)
        base = _slugify(row.prompt[:50])

    candidate = f"{base}.{ext}"
    counter = 1
    while (images_dir / candidate).exists():
        candidate = f"{base}_{counter}.{ext}"
        counter += 1

    return candidate


def _slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    import re
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text.strip("_")
    return text or "image"
