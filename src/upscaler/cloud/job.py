"""High-level cloud job lifecycle orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from upscaler.cloud.transfer import DataTransfer
from upscaler.cloud.vast import DEFAULT_IMAGE, VastClient, VastInstance
from upscaler.core.checkpoint import JobManifest

logger = logging.getLogger(__name__)

# Remote paths on the vast.ai instance
REMOTE_DATA = "/data"
REMOTE_INPUT = f"{REMOTE_DATA}/input"
REMOTE_OUTPUT = f"{REMOTE_DATA}/output"
REMOTE_WORK = f"{REMOTE_DATA}/work"
REMOTE_MANIFEST = f"{REMOTE_WORK}/manifest.json"


class CloudJob:
    """Manages the full lifecycle of a cloud upscaling job.

    Coordinates instance creation, data transfer, remote execution,
    and result retrieval for vast.ai GPU instances.
    """

    def __init__(
        self,
        manifest: JobManifest,
        gpu_filter: str = "RTX_3090",
        image: str = DEFAULT_IMAGE,
        interruptible: bool = True,
        max_price: float | None = None,
        vast_client: VastClient | None = None,
    ) -> None:
        self._manifest = manifest
        self._gpu_filter = gpu_filter
        self._image = image
        self._interruptible = interruptible
        self._max_price = max_price
        self._vast = vast_client or VastClient()
        self._instance: VastInstance | None = None
        self._transfer: DataTransfer | None = None

    # -- public API --------------------------------------------------------

    def submit(
        self,
        input_path: Path,
        preset: str | None = None,
        segment_range: tuple[int, int] | None = None,
    ) -> str:
        """Find cheapest instance, upload data, start processing.

        Args:
            input_path: Local path to the input video.
            preset: Optional hardware preset for remote processing.
            segment_range: Optional (start, end) segment indices. When set,
                the remote upscaler only processes segments in [start, end).

        Returns:
            The job ID.
        """
        # 1. Find cheapest offer
        offers = self._vast.search_offers(
            gpu_filter=self._gpu_filter,
            interruptible=self._interruptible,
            max_price_per_hour=self._max_price,
        )
        if not offers:
            raise RuntimeError(
                f"No offers found for {self._gpu_filter}"
                + (f" under ${self._max_price}/hr" if self._max_price else "")
            )

        offer = offers[0]
        price = offer.get("dph_total", 0)
        logger.info(
            "Best offer: #%s at $%.3f/hr (%s)",
            offer["id"],
            price,
            offer.get("gpu_name", "?"),
        )

        # 2. Create instance
        instance_id = self._vast.create_instance(
            offer_id=offer["id"],
            image=self._image,
        )

        # 3. Wait for it to start
        logger.info("Waiting for instance #%s to start...", instance_id)
        self._instance = self._vast.wait_until_running(instance_id)
        self._transfer = DataTransfer(self._instance.ssh_host, self._instance.ssh_port)

        # 4. Record instance in manifest
        self._manifest._data.setdefault("instance_history", []).append(
            {
                "instance_id": instance_id,
                "gpu_name": self._instance.gpu_name,
                "price_per_hour": self._instance.price_per_hour,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        self._manifest.save()

        # 5. Upload input video and manifest
        logger.info("Uploading input video...")
        self._transfer.ssh_exec(f"mkdir -p {REMOTE_INPUT} {REMOTE_WORK}")
        self._transfer.upload(input_path, f"{REMOTE_INPUT}/{input_path.name}")
        self._transfer.upload_manifest(self._manifest.manifest_path, REMOTE_MANIFEST)

        # 6. Start remote processing
        preset_flag = f" --preset {preset}" if preset else ""
        range_flag = ""
        if segment_range is not None:
            range_flag = f" --segment-range {segment_range[0]} {segment_range[1]}"
        cmd = (
            f"nohup upscaler upscale {REMOTE_INPUT}/{input_path.name}"
            f" -o {REMOTE_OUTPUT}/output.mp4"
            f" --checkpoint"
            f"{preset_flag}"
            f"{range_flag}"
            f" -v"
            f" > {REMOTE_WORK}/upscaler.log 2>&1 &"
        )
        self._transfer.ssh_exec(cmd)
        logger.info("Remote processing started (detached)")

        return self._manifest.job_id

    def follow(self) -> None:
        """Attach to a running job and stream the log output."""
        if self._transfer is None:
            self._connect_to_instance()
        assert self._transfer is not None
        # tail -f the log, which will stream until the process ends
        self._transfer.ssh_exec(f"tail -f {REMOTE_WORK}/upscaler.log", timeout=None)

    def status(self) -> dict:
        """Check job status by downloading the manifest from the instance."""
        if self._transfer is None:
            self._connect_to_instance()
        assert self._transfer is not None

        local_tmp = self._manifest.manifest_path.parent / "remote_manifest.json"
        try:
            self._transfer.download_manifest(REMOTE_MANIFEST, local_tmp)
            remote = JobManifest.load(local_tmp)
        except Exception as exc:
            return {
                "job_id": self._manifest.job_id,
                "status": "unreachable",
                "error": str(exc),
            }

        completed = len(remote.completed_segments())
        total = remote.total_segments
        return {
            "job_id": remote.job_id,
            "status": "complete" if remote.is_complete() else "running",
            "progress": f"{completed}/{total} segments",
            "instance_id": (self._instance.instance_id if self._instance else None),
        }

    def resume(self, preset: str | None = None) -> str:
        """Resume a preempted job on a new instance.

        Re-uses the existing manifest to skip completed segments.
        """
        # Re-read manifest to get latest state
        self._manifest = JobManifest.load(self._manifest.manifest_path)
        if self._manifest.is_complete():
            logger.info("Job already complete — nothing to resume")
            return self._manifest.job_id

        input_file = Path(self._manifest._data["input_file"])
        return self.submit(input_file, preset=preset)

    def results(self, output_dir: Path) -> Path:
        """Download completed results from the instance."""
        if self._transfer is None:
            self._connect_to_instance()
        assert self._transfer is not None

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading results to %s ...", output_dir)
        self._transfer.download(f"{REMOTE_OUTPUT}/", output_dir)

        # Also download the final manifest
        self._transfer.download_manifest(
            REMOTE_MANIFEST,
            output_dir / "manifest.json",
        )

        return output_dir

    def cancel(self) -> None:
        """Destroy the instance and update the manifest."""
        if self._instance is not None:
            self._vast.destroy_instance(self._instance.instance_id)
        self._manifest.save()
        logger.info("Job %s cancelled", self._manifest.job_id)

    # -- internals ---------------------------------------------------------

    def _connect_to_instance(self) -> None:
        """Connect to the current instance from manifest history."""
        history = self._manifest._data.get("instance_history", [])
        if not history:
            raise RuntimeError("No instance history — submit a job first")
        last = history[-1]
        instance_id = last["instance_id"]
        self._instance = self._vast.get_instance(instance_id)
        self._transfer = DataTransfer(self._instance.ssh_host, self._instance.ssh_port)
