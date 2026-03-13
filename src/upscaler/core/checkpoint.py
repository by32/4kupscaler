"""Job manifest and checkpoint management for resumable segmented processing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SegmentStatus:
    """Tracks the processing state of a single video segment."""

    index: int
    start_frame: int
    end_frame: int
    status: str = "pending"  # pending | in_progress | completed
    output_file: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None


class JobManifest:
    """Persistent job state for resumable processing.

    The manifest is the single source of truth for which segments need
    processing.  It is saved atomically (write-to-tmp then rename) to
    survive unexpected interruptions.

    Usage::

        # Create a new job
        manifest = JobManifest.create(path, input_file, config, segments)

        # Resume an existing job
        manifest = JobManifest.load(path)
        while (idx := manifest.next_pending_segment()) is not None:
            manifest.mark_segment_started(idx)
            # ... process segment ...
            manifest.mark_segment_completed(idx, output_file, duration)
            manifest.save()
    """

    def __init__(self, manifest_path: Path, data: dict[str, Any]) -> None:
        self._path = manifest_path
        self._data = data
        self.segments: list[SegmentStatus] = [
            SegmentStatus(**s) for s in data["segments"]
        ]

    # -- constructors ------------------------------------------------------

    @classmethod
    def create(
        cls,
        manifest_path: Path,
        input_file: Path,
        config: object,
        segments: list[tuple[int, int]],
    ) -> JobManifest:
        """Create a new manifest for a fresh job."""
        seg_list = [
            asdict(
                SegmentStatus(
                    index=i,
                    start_frame=start,
                    end_frame=end,
                )
            )
            for i, (start, end) in enumerate(segments)
        ]
        data: dict[str, Any] = {
            "job_id": uuid.uuid4().hex[:12],
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_file": str(input_file),
            "input_sha256": _quick_hash(input_file),
            "config": _serialize_config(config),
            "segments": seg_list,
            "instance_history": [],
            "total_cost_usd": 0.0,
        }
        manifest = cls(manifest_path, data)
        manifest.save()
        logger.info(
            "Created job manifest %s (%d segments)", manifest.job_id, len(seg_list)
        )
        return manifest

    @classmethod
    def load(cls, manifest_path: Path) -> JobManifest:
        """Load an existing manifest from JSON."""
        with open(manifest_path) as f:
            data = json.load(f)
        manifest = cls(manifest_path, data)
        pending = sum(1 for s in manifest.segments if s.status == "pending")
        logger.info(
            "Loaded job %s: %d/%d segments pending",
            manifest.job_id,
            pending,
            len(manifest.segments),
        )
        return manifest

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        """Atomically write manifest to disk."""
        self._data["segments"] = [asdict(s) for s in self.segments]
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp_path, self._path)

    # -- segment management ------------------------------------------------

    def mark_segment_started(self, seg_idx: int) -> None:
        self.segments[seg_idx].status = "in_progress"

    def mark_segment_completed(
        self, seg_idx: int, output_file: str, duration_s: float
    ) -> None:
        seg = self.segments[seg_idx]
        seg.status = "completed"
        seg.output_file = output_file
        seg.completed_at = datetime.now(timezone.utc).isoformat()
        seg.duration_s = round(duration_s, 2)

    def next_pending_segment(self) -> int | None:
        """Return index of next pending segment, or None if all done."""
        for seg in self.segments:
            if seg.status == "pending":
                return seg.index
        return None

    def completed_segments(self) -> list[SegmentStatus]:
        return [s for s in self.segments if s.status == "completed"]

    def is_complete(self) -> bool:
        return all(s.status == "completed" for s in self.segments)

    # -- properties --------------------------------------------------------

    @property
    def job_id(self) -> str:
        return self._data["job_id"]

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def manifest_path(self) -> Path:
        return self._path

    def to_dict(self) -> dict[str, Any]:
        self._data["segments"] = [asdict(s) for s in self.segments]
        return dict(self._data)


# -- helpers ---------------------------------------------------------------


def _quick_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of the first 1 MB for fast identity checks."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(chunk_size))
    except OSError:
        return ""
    return h.hexdigest()


def _serialize_config(config: object) -> dict[str, Any]:
    """Serialize an UpscaleConfig to a JSON-safe dict."""
    if hasattr(config, "model_dump"):
        data = config.model_dump(mode="json")
    elif hasattr(config, "dict"):
        data = config.dict()
    else:
        data = {}
    # Convert Path objects to strings
    for key, val in data.items():
        if isinstance(val, Path):
            data[key] = str(val)
    return data
