"""Multi-GPU fleet orchestrator — split one video across N vast.ai instances."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from upscaler.cloud.transfer import DataTransfer
from upscaler.cloud.vast import DEFAULT_IMAGE, VastClient, VastInstance
from upscaler.core.checkpoint import JobManifest

logger = logging.getLogger(__name__)

# Remote paths (same as cloud/job.py)
REMOTE_DATA = "/data"
REMOTE_INPUT = f"{REMOTE_DATA}/input"
REMOTE_OUTPUT = f"{REMOTE_DATA}/output"
REMOTE_WORK = f"{REMOTE_DATA}/work"
REMOTE_MANIFEST = f"{REMOTE_WORK}/manifest.json"


class FleetJob:
    """Coordinate N parallel workers processing disjoint segment ranges.

    Each worker is a vast.ai instance running the upscaler with
    ``--segment-range START END`` so it only processes its assigned slice.
    Segment files are downloaded from each worker and merged locally.

    Usage::

        manifest = JobManifest.create(...)
        fleet = FleetJob(manifest, num_workers=4)
        fleet.submit(input_path)
        fleet.wait()
        fleet.gather(output_dir)
    """

    def __init__(
        self,
        manifest: JobManifest,
        num_workers: int = 3,
        gpu_filter: str = "RTX_3090",
        image: str = DEFAULT_IMAGE,
        interruptible: bool = True,
        max_price: float | None = None,
        vast_client: VastClient | None = None,
    ) -> None:
        self._manifest = manifest
        self._num_workers = min(num_workers, manifest.total_segments)
        self._gpu_filter = gpu_filter
        self._image = image
        self._interruptible = interruptible
        self._max_price = max_price
        self._vast = vast_client or VastClient()
        self._instances: dict[int, VastInstance] = {}
        self._transfers: dict[int, DataTransfer] = {}

    # -- public API --------------------------------------------------------

    def submit(self, input_path: Path, preset: str | None = None) -> str:
        """Spin up N instances, upload video, start processing on each.

        Returns:
            The job ID.
        """
        # 1. Assign segments across workers
        workers = self._manifest.assign_workers(self._num_workers)
        logger.info(
            "Fleet: %d workers for %d segments",
            len(workers),
            self._manifest.total_segments,
        )

        # 2. Acquire instances in parallel
        offers = self._vast.search_offers(
            gpu_filter=self._gpu_filter,
            interruptible=self._interruptible,
            max_price_per_hour=self._max_price,
        )
        if len(offers) < len(workers):
            raise RuntimeError(
                f"Only {len(offers)} offers available, need {len(workers)}"
            )

        self._provision_workers(workers, offers)

        # 3. Upload and start on each worker in parallel
        self._start_workers(workers, input_path, preset)

        return self._manifest.job_id

    def status(self) -> dict[str, Any]:
        """Aggregate progress across all workers."""
        completed = len(self._manifest.completed_segments())
        total = self._manifest.total_segments

        worker_statuses = []
        for w in self._manifest.workers:
            wid = w["worker_id"]
            seg_start, seg_end = w["segment_start"], w["segment_end"]
            done = sum(
                1
                for i in range(seg_start, seg_end)
                if self._manifest.segments[i].status == "completed"
            )
            worker_statuses.append(
                {
                    "worker_id": wid,
                    "instance_id": w.get("instance_id"),
                    "segments": f"{done}/{seg_end - seg_start}",
                    "status": w.get("status", "unknown"),
                }
            )

        return {
            "job_id": self._manifest.job_id,
            "status": "complete" if self._manifest.is_complete() else "running",
            "progress": f"{completed}/{total} segments",
            "workers": worker_statuses,
        }

    def poll_status(self) -> dict[str, Any]:
        """Download manifest from each worker and merge progress."""
        for w in self._manifest.workers:
            wid = w["worker_id"]
            if w.get("status") in ("completed", "failed"):
                continue
            transfer = self._transfers.get(wid)
            if transfer is None:
                continue
            try:
                local_tmp = self._manifest.manifest_path.parent / f"remote_{wid}.json"
                transfer.download_manifest(REMOTE_MANIFEST, local_tmp)
                remote = JobManifest.load(local_tmp)
                # Merge segment statuses from this worker
                seg_start, seg_end = w["segment_start"], w["segment_end"]
                for i in range(seg_start, seg_end):
                    if remote.segments[i].status == "completed":
                        self._manifest.segments[i] = remote.segments[i]
                self._manifest.save()

                if self._manifest.worker_complete(wid):
                    self._manifest.set_worker_status(wid, "completed")
            except Exception as exc:
                logger.warning("Failed to poll worker %d: %s", wid, exc)

        return self.status()

    def wait(self, poll_interval: float = 30.0) -> None:
        """Block until all segments are complete, polling workers."""
        while not self._manifest.is_complete():
            self.poll_status()
            completed = len(self._manifest.completed_segments())
            total = self._manifest.total_segments
            logger.info("Fleet progress: %d/%d segments", completed, total)

            # Check for failed workers and auto-resume
            for w in self._manifest.workers:
                if w.get("status") == "running":
                    wid = w["worker_id"]
                    instance_id = w.get("instance_id")
                    if instance_id is not None:
                        try:
                            inst = self._vast.get_instance(instance_id)
                            if inst.status not in ("running",):
                                logger.warning(
                                    "Worker %d instance %d no longer running "
                                    "(status: %s) — marking failed",
                                    wid,
                                    instance_id,
                                    inst.status,
                                )
                                self._manifest.set_worker_status(wid, "failed")
                        except Exception:
                            self._manifest.set_worker_status(wid, "failed")

            if self._manifest.is_complete():
                break
            time.sleep(poll_interval)

        logger.info("Fleet: all segments complete")

    def gather(self, output_dir: Path) -> Path:
        """Download segment files from each worker and merge locally.

        Returns:
            Path to the merged output file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        seg_dir = output_dir / ".fleet_segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Download segment files from each worker
        for w in self._manifest.workers:
            wid = w["worker_id"]
            transfer = self._transfers.get(wid)
            if transfer is None:
                self._connect_worker(w)
                transfer = self._transfers.get(wid)
            if transfer is None:
                logger.warning("Cannot connect to worker %d, skipping", wid)
                continue

            seg_start, seg_end = w["segment_start"], w["segment_end"]
            for i in range(seg_start, seg_end):
                seg = self._manifest.segments[i]
                if seg.output_file:
                    remote_seg = (
                        f"{REMOTE_OUTPUT}/../work/.output_segments/{seg.output_file}"
                    )
                    local_seg = seg_dir / seg.output_file
                    try:
                        transfer.download(remote_seg, local_seg)
                    except Exception as exc:
                        logger.warning(
                            "Failed to download segment %d from worker %d: %s",
                            i,
                            wid,
                            exc,
                        )

        # Merge segments
        from upscaler.core.video_io import get_video_meta, merge_video_segments

        seg_files = sorted(seg_dir.glob("seg_*.mp4"))
        if not seg_files:
            raise RuntimeError("No segment files downloaded")

        input_path = Path(self._manifest._data["input_file"])
        meta = get_video_meta(input_path)
        output_file = output_dir / f"{input_path.stem}_upscaled.mp4"
        merge_video_segments(seg_files, output_file, meta.fps)
        logger.info("Merged %d segments into %s", len(seg_files), output_file)
        return output_file

    def cancel(self) -> None:
        """Destroy all worker instances."""
        for w in self._manifest.workers:
            instance_id = w.get("instance_id")
            if instance_id is not None:
                try:
                    self._vast.destroy_instance(instance_id)
                    logger.info(
                        "Destroyed instance %d (worker %d)", instance_id, w["worker_id"]
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to destroy instance %d: %s", instance_id, exc
                    )
                self._manifest.set_worker_status(w["worker_id"], "cancelled")
        self._manifest.save()

    # -- internals ---------------------------------------------------------

    def _provision_workers(
        self, workers: list[dict[str, Any]], offers: list[dict]
    ) -> None:
        """Create instances for all workers, waiting for each to start."""

        def _provision_one(worker: dict, offer: dict) -> tuple[int, VastInstance]:
            wid = worker["worker_id"]
            instance_id = self._vast.create_instance(
                offer_id=offer["id"], image=self._image
            )
            logger.info(
                "Worker %d: created instance %d ($%.3f/hr, %s)",
                wid,
                instance_id,
                offer.get("dph_total", 0),
                offer.get("gpu_name", "?"),
            )
            instance = self._vast.wait_until_running(instance_id)
            return wid, instance

        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = {
                pool.submit(_provision_one, w, offers[i]): w
                for i, w in enumerate(workers)
            }
            for future in as_completed(futures):
                w = futures[future]
                wid = w["worker_id"]
                try:
                    _, instance = future.result()
                    self._instances[wid] = instance
                    self._transfers[wid] = DataTransfer(
                        instance.ssh_host, instance.ssh_port
                    )
                    self._manifest.set_worker_instance(wid, instance.instance_id)
                    self._manifest._data.setdefault("instance_history", []).append(
                        {
                            "instance_id": instance.instance_id,
                            "worker_id": wid,
                            "gpu_name": instance.gpu_name,
                            "price_per_hour": instance.price_per_hour,
                            "started_at": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        }
                    )
                except Exception as exc:
                    logger.error("Worker %d: provisioning failed: %s", wid, exc)
                    self._manifest.set_worker_status(wid, "failed")

        self._manifest.save()

    def _start_workers(
        self,
        workers: list[dict[str, Any]],
        input_path: Path,
        preset: str | None,
    ) -> None:
        """Upload data and start upscaler on each worker in parallel."""

        def _start_one(worker: dict) -> None:
            wid = worker["worker_id"]
            transfer = self._transfers.get(wid)
            if transfer is None:
                logger.warning("Worker %d has no connection, skipping", wid)
                return

            seg_start = worker["segment_start"]
            seg_end = worker["segment_end"]

            # Upload input and manifest
            transfer.ssh_exec(f"mkdir -p {REMOTE_INPUT} {REMOTE_WORK} {REMOTE_OUTPUT}")
            transfer.upload(input_path, f"{REMOTE_INPUT}/{input_path.name}")
            transfer.upload_manifest(self._manifest.manifest_path, REMOTE_MANIFEST)

            # Build remote command
            preset_flag = f" --preset {preset}" if preset else ""
            cmd = (
                f"nohup upscaler upscale {REMOTE_INPUT}/{input_path.name}"
                f" -o {REMOTE_OUTPUT}/output.mp4"
                f" --checkpoint"
                f" --segment-range {seg_start} {seg_end}"
                f"{preset_flag}"
                f" -v"
                f" > {REMOTE_WORK}/upscaler_w{wid}.log 2>&1 &"
            )
            transfer.ssh_exec(cmd)
            logger.info(
                "Worker %d: started (segments [%d:%d))", wid, seg_start, seg_end
            )

        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = [pool.submit(_start_one, w) for w in workers]
            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    logger.error("Worker start failed: %s", exc)

    def _connect_worker(self, worker: dict) -> None:
        """Connect to a worker's instance."""
        instance_id = worker.get("instance_id")
        if instance_id is None:
            return
        wid = worker["worker_id"]
        try:
            instance = self._vast.get_instance(instance_id)
            self._instances[wid] = instance
            self._transfers[wid] = DataTransfer(instance.ssh_host, instance.ssh_port)
        except Exception as exc:
            logger.warning("Cannot connect to worker %d: %s", wid, exc)
