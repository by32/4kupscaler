"""Tests for job manifest and checkpoint management."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from upscaler.core.checkpoint import JobManifest, SegmentStatus


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.json"


@pytest.fixture()
def sample_segments() -> list[tuple[int, int]]:
    return [(0, 5), (5, 10), (10, 15)]


@pytest.fixture()
def mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model_dump.return_value = {
        "model": "3b-fp8",
        "resolution": 1072,
        "segment_size": 5,
    }
    return cfg


@pytest.fixture()
def input_file(tmp_path: Path) -> Path:
    f = tmp_path / "test.mp4"
    f.write_bytes(b"fake video data" * 100)
    return f


class TestSegmentStatus:
    def test_default_status(self):
        seg = SegmentStatus(index=0, start_frame=0, end_frame=5)
        assert seg.status == "pending"
        assert seg.output_file is None
        assert seg.duration_s is None


class TestJobManifest:
    def test_create_and_save(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        assert manifest_path.exists()
        assert len(manifest.segments) == 3
        assert all(s.status == "pending" for s in manifest.segments)
        assert manifest.job_id  # non-empty
        assert manifest.total_segments == 3

    def test_load_roundtrip(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        original = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        loaded = JobManifest.load(manifest_path)
        assert loaded.job_id == original.job_id
        assert loaded.total_segments == 3
        assert len(loaded.segments) == 3

    def test_mark_segment_started(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        manifest.mark_segment_started(0)
        assert manifest.segments[0].status == "in_progress"

    def test_mark_segment_completed(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        manifest.mark_segment_completed(0, "seg_0000.mp4", 120.5)
        seg = manifest.segments[0]
        assert seg.status == "completed"
        assert seg.output_file == "seg_0000.mp4"
        assert seg.duration_s == 120.5
        assert seg.completed_at is not None

    def test_next_pending_segment(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        assert manifest.next_pending_segment() == 0

        manifest.mark_segment_completed(0, "seg_0000.mp4", 100.0)
        assert manifest.next_pending_segment() == 1

        manifest.mark_segment_completed(1, "seg_0001.mp4", 100.0)
        assert manifest.next_pending_segment() == 2

        manifest.mark_segment_completed(2, "seg_0002.mp4", 100.0)
        assert manifest.next_pending_segment() is None

    def test_is_complete(self, manifest_path, sample_segments, mock_config, input_file):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        assert not manifest.is_complete()

        for i in range(3):
            manifest.mark_segment_completed(i, f"seg_{i:04d}.mp4", 100.0)
        assert manifest.is_complete()

    def test_completed_segments(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        assert len(manifest.completed_segments()) == 0

        manifest.mark_segment_completed(0, "seg_0000.mp4", 100.0)
        manifest.mark_segment_completed(2, "seg_0002.mp4", 100.0)
        assert len(manifest.completed_segments()) == 2

    def test_atomic_save(self, manifest_path, sample_segments, mock_config, input_file):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        manifest.mark_segment_completed(0, "seg_0000.mp4", 100.0)
        manifest.save()

        # Verify no temp file left behind
        assert not manifest_path.with_suffix(".tmp").exists()

        # Verify content is valid JSON with updated state
        data = json.loads(manifest_path.read_text())
        assert data["segments"][0]["status"] == "completed"

    def test_resume_partial_job(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        """Simulate: create job, complete 2 segments, save, reload, resume."""
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        manifest.mark_segment_completed(0, "seg_0000.mp4", 100.0)
        manifest.mark_segment_completed(1, "seg_0001.mp4", 110.0)
        manifest.save()

        # Simulate restart by loading from disk
        resumed = JobManifest.load(manifest_path)
        assert not resumed.is_complete()
        assert resumed.next_pending_segment() == 2
        assert len(resumed.completed_segments()) == 2

    def test_to_dict(self, manifest_path, sample_segments, mock_config, input_file):
        manifest = JobManifest.create(
            manifest_path, input_file, mock_config, sample_segments
        )
        d = manifest.to_dict()
        assert "job_id" in d
        assert "segments" in d
        assert len(d["segments"]) == 3

    def test_input_hash_stored(
        self, manifest_path, sample_segments, mock_config, input_file
    ):
        JobManifest.create(manifest_path, input_file, mock_config, sample_segments)
        data = json.loads(manifest_path.read_text())
        assert "input_sha256" in data
        assert len(data["input_sha256"]) == 64  # SHA-256 hex
