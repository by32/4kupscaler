"""Tests for FleetJob multi-GPU orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from upscaler.cloud.fleet import FleetJob
from upscaler.core.checkpoint import JobManifest


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.json"


@pytest.fixture()
def segments() -> list[tuple[int, int]]:
    return [(i * 5, (i + 1) * 5) for i in range(12)]


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
def manifest(manifest_path, segments, mock_config, input_file):
    return JobManifest.create(manifest_path, input_file, mock_config, segments)


def _make_vast_client(num_offers: int = 4) -> MagicMock:
    client = MagicMock()
    client.search_offers.return_value = [
        {"id": i + 100, "gpu_name": "RTX 3090", "dph_total": 0.25 + i * 0.01}
        for i in range(num_offers)
    ]
    client.create_instance.side_effect = lambda **kw: kw["offer_id"] + 1000
    client.wait_until_running.side_effect = lambda iid, **kw: MagicMock(
        instance_id=iid,
        gpu_name="RTX 3090",
        ssh_host="1.2.3.4",
        ssh_port=22000 + iid,
        price_per_hour=0.25,
        status="running",
    )
    client.get_instance.side_effect = lambda iid: MagicMock(
        instance_id=iid,
        gpu_name="RTX 3090",
        ssh_host="1.2.3.4",
        ssh_port=22000 + iid,
        price_per_hour=0.25,
        status="running",
    )
    return client


class TestFleetJobInit:
    def test_caps_workers_to_segments(self, manifest):
        fleet = FleetJob(manifest=manifest, num_workers=100)
        assert fleet._num_workers == 12

    def test_default_workers(self, manifest):
        fleet = FleetJob(manifest=manifest)
        assert fleet._num_workers == 3


class TestFleetSubmit:
    @patch("upscaler.cloud.fleet.DataTransfer")
    def test_submit_provisions_workers(self, mock_dt_cls, manifest, input_file):
        mock_dt = MagicMock()
        mock_dt_cls.return_value = mock_dt

        client = _make_vast_client(4)
        fleet = FleetJob(manifest=manifest, num_workers=3, vast_client=client)
        job_id = fleet.submit(input_file)

        assert job_id == manifest.job_id
        assert len(manifest.workers) == 3
        assert client.create_instance.call_count == 3
        assert client.wait_until_running.call_count == 3

    @patch("upscaler.cloud.fleet.DataTransfer")
    def test_submit_starts_remote_commands(self, mock_dt_cls, manifest, input_file):
        mock_dt = MagicMock()
        mock_dt_cls.return_value = mock_dt

        client = _make_vast_client(3)
        fleet = FleetJob(manifest=manifest, num_workers=3, vast_client=client)
        fleet.submit(input_file)

        # Each worker should have had ssh_exec called (mkdir + upload + start)
        assert mock_dt.ssh_exec.call_count >= 3  # at least one per worker

    @patch("upscaler.cloud.fleet.DataTransfer")
    def test_submit_segment_ranges_in_commands(self, mock_dt_cls, manifest, input_file):
        mock_dt = MagicMock()
        mock_dt_cls.return_value = mock_dt

        client = _make_vast_client(3)
        fleet = FleetJob(manifest=manifest, num_workers=3, vast_client=client)
        fleet.submit(input_file)

        # Check that --segment-range appears in the ssh_exec calls
        all_cmds = [call.args[0] for call in mock_dt.ssh_exec.call_args_list]
        range_cmds = [c for c in all_cmds if "--segment-range" in c]
        assert len(range_cmds) == 3

    @patch("upscaler.cloud.fleet.DataTransfer")
    def test_submit_not_enough_offers(self, mock_dt_cls, manifest, input_file):
        client = _make_vast_client(1)
        fleet = FleetJob(manifest=manifest, num_workers=3, vast_client=client)
        with pytest.raises(RuntimeError, match="Only 1 offers"):
            fleet.submit(input_file)


class TestFleetStatus:
    def test_status_all_pending(self, manifest):
        manifest.assign_workers(3)
        fleet = FleetJob(manifest=manifest)
        info = fleet.status()
        assert info["status"] == "running"
        assert info["progress"] == "0/12 segments"
        assert len(info["workers"]) == 3

    def test_status_partially_complete(self, manifest):
        manifest.assign_workers(3)
        # Complete first worker's segments
        for i in range(4):
            manifest.mark_segment_completed(i, f"seg_{i:04d}.mp4", 10.0)
        fleet = FleetJob(manifest=manifest)
        info = fleet.status()
        assert info["progress"] == "4/12 segments"

    def test_status_all_complete(self, manifest):
        manifest.assign_workers(3)
        for i in range(12):
            manifest.mark_segment_completed(i, f"seg_{i:04d}.mp4", 10.0)
        fleet = FleetJob(manifest=manifest)
        info = fleet.status()
        assert info["status"] == "complete"


class TestFleetCancel:
    @patch("upscaler.cloud.fleet.DataTransfer")
    def test_cancel_destroys_all_instances(self, mock_dt_cls, manifest, input_file):
        mock_dt = MagicMock()
        mock_dt_cls.return_value = mock_dt

        client = _make_vast_client(3)
        fleet = FleetJob(manifest=manifest, num_workers=3, vast_client=client)
        fleet.submit(input_file)

        client.destroy_instance.reset_mock()
        fleet.cancel()
        assert client.destroy_instance.call_count == 3
