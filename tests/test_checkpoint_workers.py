"""Tests for worker assignment and fleet tracking in JobManifest."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from upscaler.core.checkpoint import JobManifest


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.json"


@pytest.fixture()
def many_segments() -> list[tuple[int, int]]:
    """20 segments for fleet splitting."""
    return [(i * 5, (i + 1) * 5) for i in range(20)]


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


@pytest.fixture()
def manifest(manifest_path, many_segments, mock_config, input_file):
    return JobManifest.create(manifest_path, input_file, mock_config, many_segments)


class TestWorkerAssignment:
    def test_assign_workers_even_split(self, manifest):
        workers = manifest.assign_workers(4)
        assert len(workers) == 4
        # 20 segments / 4 workers = 5 each
        assert workers[0]["segment_start"] == 0
        assert workers[0]["segment_end"] == 5
        assert workers[1]["segment_start"] == 5
        assert workers[1]["segment_end"] == 10
        assert workers[2]["segment_start"] == 10
        assert workers[2]["segment_end"] == 15
        assert workers[3]["segment_start"] == 15
        assert workers[3]["segment_end"] == 20

    def test_assign_workers_uneven_split(self, manifest):
        workers = manifest.assign_workers(3)
        assert len(workers) == 3
        # 20 / 3 = 6 remainder 2, so first 2 workers get 7, last gets 6
        assert workers[0]["segment_end"] - workers[0]["segment_start"] == 7
        assert workers[1]["segment_end"] - workers[1]["segment_start"] == 7
        assert workers[2]["segment_end"] - workers[2]["segment_start"] == 6

    def test_assign_workers_capped_at_segment_count(
        self, manifest_path, mock_config, input_file
    ):
        """More workers than segments → cap to segment count."""
        small_segs = [(0, 5), (5, 10)]
        m = JobManifest.create(manifest_path, input_file, mock_config, small_segs)
        workers = m.assign_workers(10)
        assert len(workers) == 2

    def test_assign_workers_single(self, manifest):
        workers = manifest.assign_workers(1)
        assert len(workers) == 1
        assert workers[0]["segment_start"] == 0
        assert workers[0]["segment_end"] == 20

    def test_assign_workers_persisted(self, manifest, manifest_path):
        manifest.assign_workers(4)
        loaded = JobManifest.load(manifest_path)
        assert len(loaded.workers) == 4

    def test_all_segments_covered(self, manifest):
        workers = manifest.assign_workers(4)
        covered = set()
        for w in workers:
            for i in range(w["segment_start"], w["segment_end"]):
                covered.add(i)
        assert covered == set(range(20))


class TestSegmentsForWorker:
    def test_returns_range(self, manifest):
        manifest.assign_workers(4)
        start, end = manifest.segments_for_worker(2)
        assert start == 10
        assert end == 15

    def test_unknown_worker_raises(self, manifest):
        manifest.assign_workers(4)
        with pytest.raises(ValueError, match="Unknown worker_id"):
            manifest.segments_for_worker(99)


class TestWorkerComplete:
    def test_incomplete(self, manifest):
        manifest.assign_workers(4)
        assert not manifest.worker_complete(0)

    def test_complete(self, manifest):
        manifest.assign_workers(4)
        for i in range(5):
            manifest.mark_segment_completed(i, f"seg_{i:04d}.mp4", 10.0)
        assert manifest.worker_complete(0)
        assert not manifest.worker_complete(1)


class TestWorkerInstanceTracking:
    def test_set_worker_instance(self, manifest):
        manifest.assign_workers(2)
        manifest.set_worker_instance(0, 12345)
        assert manifest.workers[0]["instance_id"] == 12345
        assert manifest.workers[0]["status"] == "running"

    def test_set_worker_status(self, manifest):
        manifest.assign_workers(2)
        manifest.set_worker_status(1, "failed")
        assert manifest.workers[1]["status"] == "failed"

    def test_unknown_worker_instance_raises(self, manifest):
        manifest.assign_workers(2)
        with pytest.raises(ValueError):
            manifest.set_worker_instance(99, 12345)

    def test_unknown_worker_status_raises(self, manifest):
        manifest.assign_workers(2)
        with pytest.raises(ValueError):
            manifest.set_worker_status(99, "failed")
