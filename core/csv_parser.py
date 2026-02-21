"""CSV parsing and validation for batch image generation jobs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from core.config import AppConfig, VALID_ASPECT_RATIOS, VALID_MODELS, VALID_RESOLUTIONS


class PromptRow(BaseModel):
    """Represents a single validated row from the batch CSV."""

    prompt: str
    aspect_ratio: str = Field(default="1:1")
    resolution: str = Field(default="1K")
    model: str = Field(default="gemini-2.0-flash-preview-image-generation")
    group: Optional[str] = Field(default=None)
    reference_images: list[str] = Field(default_factory=list)
    output_filename: Optional[str] = Field(default=None)
    style_instructions: Optional[str] = Field(default=None)
    negative_prompt: Optional[str] = Field(default=None)

    # Row index for tracking/logging (not part of the CSV schema)
    row_index: int = Field(default=0, exclude=True)

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("'prompt' column must not be empty.")
        return v

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        v = v.strip() or "1:1"
        if v not in VALID_ASPECT_RATIOS:
            raise ValueError(
                f"Invalid aspect_ratio '{v}'. Must be one of: {', '.join(sorted(VALID_ASPECT_RATIOS))}"
            )
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        v = v.strip() or "1K"
        if v not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution '{v}'. Must be one of: {', '.join(sorted(VALID_RESOLUTIONS))}"
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return "gemini-2.0-flash-preview-image-generation"
        if v not in VALID_MODELS:
            raise ValueError(
                f"Invalid model '{v}'. Must be one of: {', '.join(sorted(VALID_MODELS))}"
            )
        return v

    @field_validator("reference_images", mode="before")
    @classmethod
    def parse_reference_images(cls, v: object) -> list[str]:
        """Parse semicolon-separated paths from a string, or pass through a list."""
        if isinstance(v, list):
            return [str(p).strip() for p in v if str(p).strip()]
        if not v:
            return []
        return [p.strip() for p in str(v).split(";") if p.strip()]


class ParsedCSV(BaseModel):
    """Result of parsing and validating the entire CSV file."""

    rows: list[PromptRow]
    errors: list[str] = Field(default_factory=list)


def parse_csv(csv_path: Path, config: AppConfig) -> ParsedCSV:
    """
    Read and validate a batch prompts CSV file.

    Applies config defaults where CSV values are blank. Validates reference image
    paths exist on disk. Collects per-row errors without raising immediately so
    callers can decide whether to abort or skip bad rows.

    Args:
        csv_path: Path to the CSV file.
        config: Loaded AppConfig used for defaults and reference group resolution.

    Returns:
        ParsedCSV with valid rows and any per-row error messages.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is missing required columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file appears to be empty.")

        fieldnames_lower = [fn.strip().lower() for fn in reader.fieldnames]
        if "prompt" not in fieldnames_lower:
            raise ValueError(
                "CSV is missing the required 'prompt' column. "
                f"Found columns: {list(reader.fieldnames)}"
            )

        rows: list[PromptRow] = []
        errors: list[str] = []

        for line_num, raw_row in enumerate(reader, start=2):  # start=2: header is line 1
            # Normalize column names (strip whitespace, lowercase).
            # csv.DictReader uses key None for surplus columns; skip them.
            row: dict[str, str] = {}
            # csv.DictReader puts all values for columns beyond the header into
            # a list stored under the key None — handle it explicitly before
            # iterating so we don't try to call None.strip() below.
            extra_values = raw_row.get(None)  # type: ignore[call-overload]
            if extra_values:
                errors.append(
                    f"Row {line_num}: Extra columns detected with values {extra_values!r}."
                )
            for k, v in raw_row.items():
                if k is None:
                    continue
                row[k.strip().lower()] = (v or "").strip()

            # Apply config defaults for optional fields
            row.setdefault("aspect_ratio", config.default_aspect_ratio)
            row.setdefault("resolution", config.default_resolution)
            row.setdefault("model", config.default_model)

            if not row.get("aspect_ratio"):
                row["aspect_ratio"] = config.default_aspect_ratio
            if not row.get("resolution"):
                row["resolution"] = config.default_resolution
            if not row.get("model"):
                row["model"] = config.default_model

            try:
                prompt_row = PromptRow(row_index=line_num, **row)
            except Exception as exc:
                errors.append(f"Row {line_num}: Validation error — {exc}")
                continue

            # Validate per-prompt reference image paths; skip row on any error
            ref_path_errors = _validate_ref_paths(
                prompt_row.reference_images, line_num, label="reference_images"
            )
            if ref_path_errors:
                errors.extend(ref_path_errors)
                continue

            # Resolve and validate group reference image paths; skip row on any error
            if prompt_row.group:
                if prompt_row.group not in config.reference_groups:
                    errors.append(
                        f"Row {line_num}: Unknown group '{prompt_row.group}'. "
                        "Add it to reference_groups in config.yaml."
                    )
                    continue
                group_refs = config.reference_groups[prompt_row.group]
                group_path_errors = _validate_ref_paths(
                    group_refs, line_num, label=f"group '{prompt_row.group}'"
                )
                if group_path_errors:
                    errors.extend(group_path_errors)
                    continue

            rows.append(prompt_row)

    return ParsedCSV(rows=rows, errors=errors)


def _validate_ref_paths(paths: list[str], line_num: int, label: str) -> list[str]:
    """Return error messages for any reference image paths that don't exist."""
    errors = []
    for path_str in paths:
        if not Path(path_str).exists():
            errors.append(
                f"Row {line_num}: {label} path not found: '{path_str}'"
            )
    return errors


def get_effective_reference_images(row: PromptRow, config: AppConfig) -> list[str]:
    """
    Return the merged list of reference image paths for a given prompt row.

    Group references come first, then per-prompt references (deduped, max 14 total).

    Args:
        row: The validated PromptRow.
        config: AppConfig containing reference_groups.

    Returns:
        List of file path strings (at most 14 entries).
    """
    seen: set[str] = set()
    combined: list[str] = []

    group_refs = config.reference_groups.get(row.group or "", []) if row.group else []

    for path_str in group_refs + row.reference_images:
        if path_str not in seen:
            seen.add(path_str)
            combined.append(path_str)

    return combined[:14]  # Gemini supports up to 14 reference images
