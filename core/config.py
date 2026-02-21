"""Configuration loading, validation, and environment variable support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


VALID_MODELS = {
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-preview-05-20",
    "gemini-3-pro-image-preview",
}

VALID_ASPECT_RATIOS = {"1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "4:5", "5:4"}
VALID_RESOLUTIONS = {"1K", "2K", "4K"}
VALID_OUTPUT_FORMATS = {"png", "jpeg", "webp"}


class AppConfig(BaseModel):
    """Main application configuration loaded from config.yaml or environment variables."""

    gemini_api_key: str = Field(default="")
    default_model: str = Field(default="gemini-2.0-flash-preview-image-generation")
    default_aspect_ratio: str = Field(default="1:1")
    default_resolution: str = Field(default="1K")
    output_directory: str = Field(default="./output")
    output_format: str = Field(default="png")
    delay_between_requests: float = Field(default=3.0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    generate_thumbnails: bool = Field(default=True)
    thumbnail_size: int = Field(default=256, ge=32)
    reference_groups: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("default_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in VALID_MODELS:
            raise ValueError(
                f"Invalid default_model '{v}'. Must be one of: {', '.join(sorted(VALID_MODELS))}"
            )
        return v

    @field_validator("default_aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in VALID_ASPECT_RATIOS:
            raise ValueError(
                f"Invalid default_aspect_ratio '{v}'. Must be one of: {', '.join(sorted(VALID_ASPECT_RATIOS))}"
            )
        return v

    @field_validator("default_resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        if v not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid default_resolution '{v}'. Must be one of: {', '.join(sorted(VALID_RESOLUTIONS))}"
            )
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        v = v.lower()
        if v not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{v}'. Must be one of: {', '.join(sorted(VALID_OUTPUT_FORMATS))}"
            )
        return v

    @model_validator(mode="after")
    def apply_env_overrides(self) -> "AppConfig":
        """Allow GEMINI_API_KEY environment variable to override config file value."""
        env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_key:
            self.gemini_api_key = env_key
        return self


def load_config(config_path: Path) -> AppConfig:
    """
    Load and validate application configuration from a YAML file.

    Environment variable GEMINI_API_KEY takes precedence over the config file value.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config contains invalid values.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Run 'cp config.yaml.example config.yaml' and fill in your API key."
        )

    with config_path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Flatten reference_groups if nested under a key
    reference_groups_raw = raw.pop("reference_groups", {}) or {}

    config = AppConfig(**raw, reference_groups=reference_groups_raw)

    # Validate reference group paths exist
    for group_name, paths in config.reference_groups.items():
        for path_str in paths:
            p = Path(path_str)
            if not p.exists():
                raise FileNotFoundError(
                    f"Reference image in group '{group_name}' not found: {path_str}"
                )

    if not config.gemini_api_key:
        raise ValueError(
            "Gemini API key is not set. Set 'gemini_api_key' in config.yaml "
            "or export the GEMINI_API_KEY environment variable."
        )

    return config
