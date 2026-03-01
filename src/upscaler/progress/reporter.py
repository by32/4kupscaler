"""Rich-based progress reporting with ETA and optional VRAM display."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


class ProgressReporter:
    """Rich-based progress display for the upscale pipeline.

    Implements the ``ProgressCallback`` signature ``(int, int, str) -> None``
    so it can be passed directly to ``UpscaleEngine.run()``.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )
        self._task_id: object | None = None
        self._started = False

    def callback(self, current_step: int, total_steps: int, phase: str) -> None:
        """Progress callback compatible with UpscaleEngine.

        Args:
            current_step: Current phase number (1-based).
            total_steps: Total number of phases.
            phase: Name of the current phase (e.g. "Encoding").
        """
        if not self._started:
            self._progress.start()
            self._task_id = self._progress.add_task(phase, total=total_steps)
            self._started = True

        self._progress.update(
            self._task_id,
            completed=current_step,
            description=phase,
        )

        if current_step >= total_steps:
            self._progress.stop()
            self._started = False

    def complete(self, output_path: Path) -> None:
        """Print a success panel with the output path."""
        if self._started:
            self._progress.stop()
            self._started = False
        self.console.print(
            Panel(
                f"[green]Output saved to:[/green] {output_path}",
                title="[bold green]Upscale Complete",
                border_style="green",
            )
        )
