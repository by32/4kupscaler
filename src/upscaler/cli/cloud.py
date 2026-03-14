"""Cloud subcommands — run upscaling jobs on vast.ai GPU instances."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from upscaler.cli.common import print_error, validate_input_path

app = typer.Typer(help="Run upscaling jobs on vast.ai cloud GPUs.")

_DEFAULT_JOB_DIR = "~/.cache/upscaler/jobs"


def _job_dir(job_dir: str) -> Path:
    return Path(job_dir).expanduser()


@app.command()
def submit(
    input_video: Annotated[str, typer.Argument(help="Path to input video.")],
    gpu: Annotated[
        str, typer.Option(help="GPU filter (e.g. 'RTX_3090', 'RTX_4090').")
    ] = "RTX_3090",
    preset: Annotated[
        str | None, typer.Option(help="Hardware preset for remote processing.")
    ] = None,
    max_price: Annotated[
        float | None, typer.Option("--max-price", help="Max $/hr for instance.")
    ] = None,
    job_dir: Annotated[
        str, typer.Option(help="Directory for job state.")
    ] = _DEFAULT_JOB_DIR,
    interruptible: Annotated[
        bool,
        typer.Option(
            "--interruptible/--on-demand",
            help="Use cheaper interruptible instances.",
        ),
    ] = True,
    follow: Annotated[
        bool,
        typer.Option("--follow", help="Stream progress after submitting."),
    ] = False,
) -> None:
    """Submit a video for cloud upscaling on vast.ai."""
    console = Console()

    try:
        input_path = validate_input_path(Path(input_video))
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    from upscaler.cli.common import resolve_config
    from upscaler.cloud.job import CloudJob
    from upscaler.core.checkpoint import JobManifest
    from upscaler.core.video_io import compute_segments, get_video_meta

    # Build config to determine segments
    try:
        cfg = resolve_config(input_path=input_path, preset=preset)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    if cfg.segment_size is None:
        # Default to preset's segment_size, or 5 as fallback
        cfg.segment_size = 5

    meta = get_video_meta(cfg.input)
    segments = compute_segments(meta.frame_count, cfg.segment_size)

    # Create job directory and manifest
    jobs = _job_dir(job_dir)
    jobs.mkdir(parents=True, exist_ok=True)

    manifest = JobManifest.create(
        manifest_path=jobs / "manifest.json",
        input_file=input_path,
        config=cfg,
        segments=segments,
    )

    # Move manifest to a job-specific directory
    job_path = jobs / manifest.job_id
    job_path.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.move(str(manifest.manifest_path), str(job_path / "manifest.json"))
    manifest = JobManifest.load(job_path / "manifest.json")

    console.print(f"Job ID: [bold]{manifest.job_id}[/bold]")
    console.print(
        f"Segments: {manifest.total_segments} | "
        f"GPU: {gpu} | "
        f"Interruptible: {interruptible}"
    )

    cloud_job = CloudJob(
        manifest=manifest,
        gpu_filter=gpu,
        interruptible=interruptible,
        max_price=max_price,
    )

    try:
        cloud_job.submit(input_path, preset=preset)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    console.print(
        Panel(
            f"Job [bold]{manifest.job_id}[/bold] submitted.\n"
            f"Check status: [dim]upscaler cloud status {manifest.job_id}[/dim]",
            title="[bold green]Submitted",
            border_style="green",
        )
    )

    if follow:
        console.print("Streaming logs (Ctrl+C to detach)...")
        try:
            cloud_job.follow()
        except KeyboardInterrupt:
            console.print("\nDetached from log stream.")


@app.command()
def status(
    job_id: Annotated[str, typer.Argument(help="Job ID to check.")],
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Check status of a cloud job."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.job import CloudJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    cloud_job = CloudJob(manifest=manifest)
    info = cloud_job.status()

    table = Table(title=f"Job {job_id}")
    table.add_column("Field")
    table.add_column("Value")
    for k, v in info.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command(name="follow")
def follow_cmd(
    job_id: Annotated[str, typer.Argument(help="Job ID to follow.")],
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Attach to a running job and stream progress live."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.job import CloudJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    cloud_job = CloudJob(manifest=manifest)
    console.print(f"Following job [bold]{job_id}[/bold] (Ctrl+C to detach)...")
    try:
        cloud_job.follow()
    except KeyboardInterrupt:
        console.print("\nDetached.")


@app.command()
def resume(
    job_id: Annotated[str, typer.Argument(help="Job ID to resume.")],
    gpu: Annotated[str, typer.Option(help="GPU filter.")] = "RTX_3090",
    preset: Annotated[str | None, typer.Option(help="Hardware preset.")] = None,
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Resume a preempted or failed cloud job on a new instance."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.job import CloudJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    completed = len(manifest.completed_segments())
    console.print(
        f"Resuming job [bold]{job_id}[/bold]: "
        f"{completed}/{manifest.total_segments} segments done"
    )

    cloud_job = CloudJob(manifest=manifest, gpu_filter=gpu)
    try:
        cloud_job.resume(preset=preset)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    console.print("[bold green]Job resumed on new instance.[/bold green]")


@app.command()
def results(
    job_id: Annotated[str, typer.Argument(help="Job ID.")],
    output: Annotated[str, typer.Option("-o", help="Output directory.")] = ".",
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Download results from a completed cloud job."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.job import CloudJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    cloud_job = CloudJob(manifest=manifest)

    out_dir = Path(output)
    try:
        result_path = cloud_job.results(out_dir)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    console.print(
        Panel(
            f"[green]Results downloaded to:[/green] {result_path}",
            title="[bold green]Download Complete",
            border_style="green",
        )
    )


@app.command()
def scatter(
    input_video: Annotated[str, typer.Argument(help="Path to input video.")],
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Number of parallel GPU instances.")
    ] = 3,
    gpu: Annotated[
        str, typer.Option(help="GPU filter (e.g. 'RTX_3090', 'RTX_4090').")
    ] = "RTX_3090",
    preset: Annotated[
        str | None, typer.Option(help="Hardware preset for remote processing.")
    ] = None,
    max_price: Annotated[
        float | None, typer.Option("--max-price", help="Max $/hr per instance.")
    ] = None,
    job_dir: Annotated[
        str, typer.Option(help="Directory for job state.")
    ] = _DEFAULT_JOB_DIR,
    interruptible: Annotated[
        bool,
        typer.Option(
            "--interruptible/--on-demand",
            help="Use cheaper interruptible instances.",
        ),
    ] = True,
) -> None:
    """Split a video across multiple GPU instances for parallel upscaling."""
    console = Console()

    try:
        input_path = validate_input_path(Path(input_video))
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    from upscaler.cli.common import resolve_config
    from upscaler.cloud.fleet import FleetJob
    from upscaler.core.checkpoint import JobManifest
    from upscaler.core.video_io import compute_segments, get_video_meta

    try:
        cfg = resolve_config(input_path=input_path, preset=preset)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    if cfg.segment_size is None:
        cfg.segment_size = 5

    meta = get_video_meta(cfg.input)
    segments = compute_segments(meta.frame_count, cfg.segment_size)

    # Create job directory and manifest
    jobs = _job_dir(job_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    manifest = JobManifest.create(
        manifest_path=jobs / "manifest.json",
        input_file=input_path,
        config=cfg,
        segments=segments,
    )

    job_path = jobs / manifest.job_id
    job_path.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.move(str(manifest.manifest_path), str(job_path / "manifest.json"))
    manifest = JobManifest.load(job_path / "manifest.json")

    effective_workers = min(workers, len(segments))
    console.print(f"Job ID: [bold]{manifest.job_id}[/bold]")
    console.print(
        f"Segments: {manifest.total_segments} | "
        f"Workers: {effective_workers} | "
        f"GPU: {gpu}"
    )

    fleet = FleetJob(
        manifest=manifest,
        num_workers=effective_workers,
        gpu_filter=gpu,
        interruptible=interruptible,
        max_price=max_price,
    )

    try:
        fleet.submit(input_path, preset=preset)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    # Show worker table
    info = fleet.status()
    table = Table(title="Fleet Workers")
    table.add_column("Worker")
    table.add_column("Instance")
    table.add_column("Segments")
    table.add_column("Status")
    for ws in info.get("workers", []):
        table.add_row(
            str(ws["worker_id"]),
            str(ws.get("instance_id", "-")),
            ws["segments"],
            ws["status"],
        )
    console.print(table)

    console.print(
        Panel(
            f"Fleet job [bold]{manifest.job_id}[/bold] submitted with"
            f" {effective_workers} workers.\n"
            f"Check status: [dim]upscaler cloud fleet-status {manifest.job_id}[/dim]\n"
            f"Gather results: [dim]upscaler cloud gather {manifest.job_id}[/dim]",
            title="[bold green]Fleet Submitted",
            border_style="green",
        )
    )


@app.command(name="fleet-status")
def fleet_status(
    job_id: Annotated[str, typer.Argument(help="Job ID to check.")],
    poll: Annotated[
        bool,
        typer.Option("--poll", help="Download latest progress from workers."),
    ] = False,
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Check status of a multi-GPU fleet job."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.fleet import FleetJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    if not manifest.workers:
        print_error("Not a fleet job (no worker assignments)")
        raise typer.Exit(code=1)

    fleet = FleetJob(manifest=manifest)

    info = fleet.poll_status() if poll else fleet.status()

    # Overall progress
    console.print(f"Job [bold]{job_id}[/bold]: {info['progress']} — {info['status']}")

    # Worker table
    table = Table(title="Workers")
    table.add_column("Worker")
    table.add_column("Instance")
    table.add_column("Segments")
    table.add_column("Status")
    for ws in info.get("workers", []):
        style = (
            "green"
            if ws["status"] == "completed"
            else "red"
            if ws["status"] == "failed"
            else None
        )
        table.add_row(
            str(ws["worker_id"]),
            str(ws.get("instance_id", "-")),
            ws["segments"],
            ws["status"],
            style=style,
        )
    console.print(table)


@app.command()
def gather(
    job_id: Annotated[str, typer.Argument(help="Job ID.")],
    output: Annotated[str, typer.Option("-o", help="Output directory.")] = ".",
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Download and merge results from a fleet job."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.fleet import FleetJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    if not manifest.workers:
        print_error("Not a fleet job — use 'results' for single-instance jobs")
        raise typer.Exit(code=1)

    fleet = FleetJob(manifest=manifest)
    out_dir = Path(output)

    try:
        result_path = fleet.gather(out_dir)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    console.print(
        Panel(
            f"[green]Merged output:[/green] {result_path}",
            title="[bold green]Fleet Gather Complete",
            border_style="green",
        )
    )


@app.command(name="fleet-cancel")
def fleet_cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID to cancel.")],
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Cancel a fleet job and destroy all worker instances."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.fleet import FleetJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    fleet = FleetJob(manifest=manifest)
    fleet.cancel()
    console.print(
        f"[bold red]Fleet job {job_id} cancelled — all instances destroyed.[/bold red]"
    )


@app.command()
def cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID to cancel.")],
    job_dir: Annotated[
        str, typer.Option(help="Job state directory.")
    ] = _DEFAULT_JOB_DIR,
) -> None:
    """Cancel a cloud job and destroy the instance."""
    console = Console()
    job_path = _job_dir(job_dir) / job_id / "manifest.json"
    if not job_path.exists():
        print_error(f"Job not found: {job_id}")
        raise typer.Exit(code=1)

    from upscaler.cloud.job import CloudJob
    from upscaler.core.checkpoint import JobManifest

    manifest = JobManifest.load(job_path)
    cloud_job = CloudJob(manifest=manifest)
    cloud_job.cancel()
    console.print(f"[bold red]Job {job_id} cancelled.[/bold red]")
