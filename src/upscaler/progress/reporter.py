"""Rich-based progress reporting with ETA and optional GPU stats display."""

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

    def __init__(
        self,
        console: Console | None = None,
        gpu_monitor: object | None = None,
    ) -> None:
        self.console = console or Console()
        self._gpu_monitor = gpu_monitor

        columns: list[object] = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ]
        if gpu_monitor is not None:
            columns.append(TextColumn("{task.fields[gpu_stats]}"))

        self._progress = Progress(*columns, console=self.console)
        self._task_id: object | None = None
        self._started = False
        self._segment_task_id: object | None = None
        self._total_segments: int = 0

    def callback(self, current_step: int, total_steps: int, phase: str) -> None:
        """Progress callback compatible with UpscaleEngine.

        Args:
            current_step: Current phase number (1-based).
            total_steps: Total number of phases.
            phase: Name of the current phase (e.g. "Encoding").
        """
        if not self._started:
            self._progress.start()
            fields = {}
            if self._gpu_monitor is not None:
                fields["gpu_stats"] = ""
            self._task_id = self._progress.add_task(phase, total=total_steps, **fields)
            self._started = True

        update_kwargs: dict[str, object] = {
            "completed": current_step,
            "description": phase,
        }
        if self._gpu_monitor is not None:
            update_kwargs["gpu_stats"] = self._format_gpu_stats()

        self._progress.update(self._task_id, **update_kwargs)

        if current_step >= total_steps:
            self._progress.stop()
            self._started = False

    # -- segmented mode ----------------------------------------------------

    def start_segmented(self, total_segments: int) -> None:
        """Initialize segment-level progress tracking."""
        self._total_segments = total_segments
        if not self._started:
            self._progress.start()
            self._started = True
        fields = {}
        if self._gpu_monitor is not None:
            fields["gpu_stats"] = ""
        self._segment_task_id = self._progress.add_task(
            f"Segment 0/{total_segments}",
            total=total_segments,
            **fields,
        )

    def begin_segment(self, seg_idx: int) -> None:
        """Mark start of a new segment."""
        if self._segment_task_id is not None:
            update_kwargs: dict[str, object] = {
                "description": f"Segment {seg_idx + 1}/{self._total_segments}",
            }
            if self._gpu_monitor is not None:
                update_kwargs["gpu_stats"] = self._format_gpu_stats()
            self._progress.update(self._segment_task_id, **update_kwargs)

    def end_segment(self) -> None:
        """Mark current segment complete."""
        if self._segment_task_id is not None:
            self._progress.advance(self._segment_task_id, 1)

    # -- completion --------------------------------------------------------

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

    # -- GPU stats ---------------------------------------------------------

    def _format_gpu_stats(self) -> str:
        """Format current GPU stats for display in the progress bar."""
        if self._gpu_monitor is None:
            return ""
        snap = self._gpu_monitor.latest()
        if snap is None:
            return ""
        vram_gb = snap.vram_used_mb / 1024
        vram_total_gb = snap.vram_total_mb / 1024
        return (
            f"[dim]{snap.temperature_c}\u00b0C | "
            f"VRAM {vram_gb:.1f}/{vram_total_gb:.1f} GB | "
            f"{snap.power_draw_w:.0f}W[/dim]"
        )
