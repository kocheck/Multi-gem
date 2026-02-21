#!/usr/bin/env python3
"""
gemini_batch.py — Batch Image Generation CLI using the Google Gemini API.

Usage:
    python gemini_batch.py --csv batch_prompts.csv [OPTIONS]

Options:
    --csv PATH          Path to the batch prompts CSV file (required)
    --config PATH       Path to config.yaml (default: config.yaml)
    --output-dir PATH   Override output directory from config
    --dry-run           Validate everything without making API calls
    --resume            Resume from a previous run (reads existing manifest)
    --preview N         Show first N rows and their parsed config, then exit
    --help              Show this message and exit
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from core.config import AppConfig, load_config
from core.csv_parser import ParsedCSV, PromptRow, parse_csv
from core.generator import ImageGenerator
from core.output import OutputManager

console = Console()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help="Path to batch prompts CSV file.",
)
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    type=click.Path(exists=False, path_type=Path),
    help="Path to config.yaml.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Override output directory from config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate CSV and config without making API calls.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from the most recent run (skips already-successful prompts).",
)
@click.option(
    "--preview",
    "preview_count",
    default=None,
    type=int,
    metavar="N",
    help="Show first N rows and parsed config, then exit.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def main(
    csv_path: Path,
    config_path: Path,
    output_dir: Optional[Path],
    dry_run: bool,
    resume: bool,
    preview_count: Optional[int],
    verbose: bool,
) -> None:
    """Batch image generation tool powered by Google Gemini."""
    _setup_logging(verbose)
    log = logging.getLogger(__name__)

    console.print(
        Panel.fit(
            "[bold magenta]Gemini Batch Image Generator[/bold magenta]\n"
            "[dim]Powered by Google Gemini API[/dim]",
            border_style="magenta",
        )
    )

    # ── Load config ──────────────────────────────────────────────────────────
    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Config validation failed:[/bold red] {exc}")
        sys.exit(1)

    if output_dir:
        config.output_directory = str(output_dir)

    # ── Parse CSV ────────────────────────────────────────────────────────────
    try:
        parsed = parse_csv(csv_path, config)
    except FileNotFoundError as exc:
        console.print(f"[bold red]CSV error:[/bold red] {exc}")
        sys.exit(1)
    except ValueError as exc:
        console.print(f"[bold red]CSV error:[/bold red] {exc}")
        sys.exit(1)

    _report_validation(parsed, csv_path)

    if parsed.errors:
        console.print(
            f"\n[bold yellow]⚠  {len(parsed.errors)} validation warning(s) found.[/bold yellow] "
            "These rows have been skipped. Review the warnings above."
        )

    if not parsed.rows:
        console.print("[bold red]No valid rows to process. Exiting.[/bold red]")
        sys.exit(1)

    # ── Preview mode ─────────────────────────────────────────────────────────
    if preview_count is not None:
        _show_preview(parsed.rows, preview_count, config)
        return

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if dry_run:
        console.print(
            f"\n[bold green]✓ Dry run complete.[/bold green] "
            f"{len(parsed.rows)} row(s) validated. No API calls made."
        )
        return

    # ── Set up output ─────────────────────────────────────────────────────────
    output_manager = OutputManager(config)

    # For --resume, find most recent run directory
    if resume:
        output_manager = _find_resume_output_manager(config)
        if output_manager is None:
            console.print("[yellow]No previous run found. Starting fresh.[/yellow]")
            output_manager = OutputManager(config)

    # Load completed entries BEFORE setup() opens the manifest for writing,
    # so we can read the existing manifest in append/resume scenarios.
    completed: set[str] = set()
    if resume:
        completed = output_manager.load_completed_filenames()

    output_manager.setup()

    # ── Determine which rows to process (skip completed if resuming) ──────────
    rows_to_process = parsed.rows
    if resume:
        if completed:
            rows_to_process = _filter_resumed_rows(parsed.rows, completed)
            skipped = len(parsed.rows) - len(rows_to_process)
            console.print(
                f"[cyan]Resuming:[/cyan] {skipped} already-completed row(s) skipped, "
                f"{len(rows_to_process)} remaining."
            )

    if not rows_to_process:
        console.print("[bold green]All rows already completed. Nothing to do![/bold green]")
        output_manager.close()
        output_manager.generate_gallery()
        return

    # ── Run the batch ─────────────────────────────────────────────────────────
    generator = ImageGenerator(config)
    success_count, error_count = _run_batch(
        rows=rows_to_process,
        generator=generator,
        output_manager=output_manager,
    )

    output_manager.close()
    output_manager.generate_gallery()

    # ── Final summary ─────────────────────────────────────────────────────────
    total = len(rows_to_process)
    console.print()
    console.print(
        Panel(
            f"[bold green]✓ {success_count}/{total} images generated successfully[/bold green]\n"
            + (f"[bold red]✗ {error_count} error(s)[/bold red]\n" if error_count else "")
            + f"\n[dim]Output:[/dim] {output_manager.run_dir}\n"
            f"[dim]Manifest:[/dim] {output_manager.manifest_path}\n"
            f"[dim]Gallery:[/dim] {output_manager.gallery_path}",
            title="Batch Complete",
            border_style="green" if error_count == 0 else "yellow",
        )
    )

    sys.exit(0 if error_count == 0 else 1)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def _run_batch(
    rows: list[PromptRow],
    generator: ImageGenerator,
    output_manager: OutputManager,
) -> tuple[int, int]:
    """
    Process all rows sequentially with a Rich progress bar.

    Returns:
        Tuple of (success_count, error_count).
    """
    success_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("Generating images...", total=len(rows))

        for row in rows:
            desc = f"[cyan]Row {row.row_index}:[/cyan] {row.prompt[:60]}…"
            progress.update(task_id, description=desc)

            result = generator.generate(row)

            if result.success and result.image_data is not None:
                filename = output_manager.save_image(
                    row, result.image_data, result.mime_type
                )
                output_manager.record_success(row, filename)
                console.print(
                    f"  [green]✓[/green] Row {row.row_index}: saved [bold]{filename}[/bold]"
                )
                success_count += 1
            else:
                error_msg = result.error or "Unknown error"
                output_manager.record_error(row, error_msg)
                console.print(
                    f"  [red]✗[/red] Row {row.row_index}: {error_msg}"
                )
                error_count += 1

            progress.advance(task_id)

    return success_count, error_count


# ---------------------------------------------------------------------------
# Helper display functions
# ---------------------------------------------------------------------------


def _report_validation(parsed: ParsedCSV, csv_path: Path) -> None:
    """Print a summary table of the parsed CSV rows."""
    console.print(
        f"\n[bold]CSV loaded:[/bold] [cyan]{csv_path}[/cyan] — "
        f"[green]{len(parsed.rows)} valid row(s)[/green]"
        + (f", [red]{len(parsed.errors)} error(s)[/red]" if parsed.errors else "")
    )

    if parsed.errors:
        for err in parsed.errors:
            console.print(f"  [red]![/red] {err}")


def _show_preview(rows: list[PromptRow], count: int, config: AppConfig) -> None:
    """Print a Rich table preview of the first N rows."""
    preview_rows = rows[:count]
    table = Table(title=f"Preview — first {len(preview_rows)} row(s)", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Prompt", max_width=50)
    table.add_column("Model", style="cyan")
    table.add_column("Aspect", style="magenta")
    table.add_column("Res", style="yellow")
    table.add_column("Group", style="blue")
    table.add_column("Ref imgs", style="green")

    for row in preview_rows:
        from core.csv_parser import get_effective_reference_images
        refs = get_effective_reference_images(row, config)
        table.add_row(
            str(row.row_index),
            row.prompt[:48] + ("…" if len(row.prompt) > 48 else ""),
            row.model.split("-")[-1],  # short model name
            row.aspect_ratio,
            row.resolution,
            row.group or "",
            str(len(refs)),
        )

    console.print(table)


def _find_resume_output_manager(config: AppConfig) -> Optional[OutputManager]:
    """Find the most recent output run directory and return an OutputManager for it."""
    base = Path(config.output_directory)
    if not base.exists():
        return None

    run_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not run_dirs:
        return None

    latest = run_dirs[0]
    return OutputManager(config, batch_timestamp=latest.name)


def _filter_resumed_rows(
    rows: list[PromptRow], completed_filenames: set[str]
) -> list[PromptRow]:
    """Return only rows whose output_filename stem is not already in completed_filenames."""
    # Compare by stem so we don't depend on the exact extension (which can
    # vary based on the API response MIME type vs config.output_format).
    completed_stems = {Path(name).stem for name in completed_filenames}

    remaining = []
    for row in rows:
        if row.output_filename and row.output_filename in completed_stems:
            continue
        remaining.append(row)
    return remaining


if __name__ == "__main__":
    main()
