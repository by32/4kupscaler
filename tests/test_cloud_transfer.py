"""Tests for data transfer module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from upscaler.cloud.transfer import DataTransfer, TransferError


@pytest.fixture()
def transfer() -> DataTransfer:
    return DataTransfer(ssh_host="host.example.com", ssh_port=12345)


class TestUpload:
    def test_builds_rsync_command(self, transfer, tmp_path):
        test_file = tmp_path / "video.mp4"
        test_file.touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            transfer.upload(test_file, "/data/input/video.mp4")
            args = mock_run.call_args[0][0]
            assert "rsync" in args
            assert str(test_file) in args
            assert "root@host.example.com:/data/input/video.mp4" in args

    def test_directory_upload_adds_trailing_slash(self, transfer, tmp_path):
        test_dir = tmp_path / "videos"
        test_dir.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            transfer.upload(test_dir, "/data/input/")
            args = mock_run.call_args[0][0]
            src = [a for a in args if str(test_dir) in a][0]
            assert src.endswith("/")


class TestDownload:
    def test_builds_rsync_command(self, transfer, tmp_path):
        output = tmp_path / "results"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            transfer.download("/data/output/", output)
            args = mock_run.call_args[0][0]
            assert "rsync" in args
            assert "root@host.example.com:/data/output/" in args


class TestSshExec:
    def test_builds_ssh_command(self, transfer):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="output", returncode=0)
            result = transfer.ssh_exec("ls /data")
            assert result == "output"
            args = mock_run.call_args[0][0]
            assert "ssh" in args
            assert "-p" in args
            assert "12345" in args
            assert "root@host.example.com" in args
            assert "ls /data" in args

    def test_raises_on_failure(self, transfer):
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ssh", stderr="error"
            )
            with pytest.raises(TransferError, match="Remote command failed"):
                transfer.ssh_exec("ls /data")

    def test_timeout(self, transfer):
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ssh", 30)
            with pytest.raises(TransferError, match="timed out"):
                transfer.ssh_exec("long command", timeout=30)
