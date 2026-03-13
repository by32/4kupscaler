"""Data transfer to/from vast.ai instances via rsync and SSH."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30"]


class TransferError(Exception):
    """Raised when a file transfer or remote command fails."""


class DataTransfer:
    """Manages file transfers and remote commands over SSH.

    Uses rsync for file transfers and ssh for command execution,
    matching the patterns from the existing ``cloud-upscale.sh`` script.
    """

    def __init__(self, ssh_host: str, ssh_port: int) -> None:
        self._host = ssh_host
        self._port = ssh_port

    # -- file transfers ----------------------------------------------------

    def upload(self, local_path: Path, remote_path: str) -> None:
        """Upload local files to the remote instance via rsync."""
        src = str(local_path)
        if local_path.is_dir():
            src = src.rstrip("/") + "/"
        dst = f"root@{self._host}:{remote_path}"
        self._rsync(src, dst)

    def download(self, remote_path: str, local_path: Path) -> None:
        """Download files from the remote instance via rsync."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        src = f"root@{self._host}:{remote_path}"
        dst = str(local_path)
        self._rsync(src, dst)

    def upload_manifest(self, manifest_path: Path, remote_path: str) -> None:
        """Upload a job manifest file to the remote instance."""
        self.upload(manifest_path, remote_path)

    def download_manifest(self, remote_path: str, local_path: Path) -> None:
        """Download a job manifest from the remote instance."""
        self.download(remote_path, local_path)

    # -- remote command execution ------------------------------------------

    def ssh_exec(self, command: str, timeout: int | None = None) -> str:
        """Execute a command on the remote instance via SSH.

        Returns:
            stdout from the remote command.

        Raises:
            TransferError: If the command fails.
        """
        cmd = [
            "ssh",
            "-p",
            str(self._port),
            *_SSH_OPTS,
            f"root@{self._host}",
            command,
        ]
        logger.debug("SSH exec: %s", command)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise TransferError(
                f"Remote command failed: {command}\n{exc.stderr}"
            ) from exc
        except subprocess.TimeoutExpired:
            raise TransferError(
                f"Remote command timed out after {timeout}s: {command}"
            ) from None
        return result.stdout

    # -- internals ---------------------------------------------------------

    def _rsync(self, src: str, dst: str) -> None:
        """Run rsync with SSH transport."""
        ssh_cmd = f"ssh -p {self._port} {' '.join(_SSH_OPTS)}"
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e",
            ssh_cmd,
            src,
            dst,
        ]
        logger.info("rsync: %s -> %s", src, dst)
        try:
            subprocess.run(cmd, check=True, timeout=3600)
        except subprocess.CalledProcessError as exc:
            raise TransferError(f"rsync failed: {exc}") from exc
