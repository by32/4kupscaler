"""Vast.ai instance lifecycle management via the vastai CLI."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "ghcr.io/by32/4kupscaler:latest"


@dataclass
class VastInstance:
    """Represents a running vast.ai instance."""

    instance_id: int
    gpu_name: str
    ssh_host: str
    ssh_port: int
    price_per_hour: float
    status: str


class VastError(Exception):
    """Raised when a vast.ai CLI command fails."""


class VastClient:
    """Wrapper around the ``vastai`` CLI for instance lifecycle management.

    Requires the ``vastai`` package to be installed and an API key
    configured via ``vastai set api-key <KEY>`` or the ``VAST_API_KEY``
    environment variable.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    # -- offer search ------------------------------------------------------

    def search_offers(
        self,
        gpu_filter: str = "RTX_3090",
        min_disk_gb: int = 40,
        min_reliability: float = 0.95,
        interruptible: bool = True,
        max_price_per_hour: float | None = None,
    ) -> list[dict]:
        """Search for available offers, sorted by price (cheapest first).

        Args:
            gpu_filter: GPU name filter (e.g. ``"RTX_3090"``).
            min_disk_gb: Minimum disk space in GB.
            min_reliability: Minimum host reliability (0-1).
            interruptible: If True, search for interruptible (spot) instances.
            max_price_per_hour: Optional maximum price filter.

        Returns:
            List of offer dicts sorted by ``dph_total``.
        """
        query_parts = [
            f"gpu_name={gpu_filter}",
            f"reliability>{min_reliability}",
            "num_gpus=1",
            "inet_down>100",
            f"disk_space>={min_disk_gb}",
        ]
        if max_price_per_hour is not None:
            query_parts.append(f"dph_total<={max_price_per_hour}")

        query = " ".join(query_parts)
        args = ["search", "offers", query, "--order", "dph_total", "--raw"]
        if interruptible:
            args.extend(["--type", "interruptible"])

        result = self._run(args)
        offers = json.loads(result)
        if not isinstance(offers, list):
            return []
        return offers

    # -- instance lifecycle ------------------------------------------------

    def create_instance(
        self,
        offer_id: int,
        image: str = DEFAULT_IMAGE,
        disk_gb: int = 40,
        onstart_cmd: str | None = None,
    ) -> int:
        """Create an instance from an offer. Returns the new instance ID."""
        args = [
            "create",
            "instance",
            str(offer_id),
            "--image",
            image,
            "--disk",
            str(disk_gb),
            "--raw",
        ]
        if onstart_cmd:
            args.extend(["--onstart-cmd", onstart_cmd])

        result = self._run(args)
        data = json.loads(result)
        instance_id = data.get("new_contract") or data.get("id")
        if not instance_id:
            raise VastError(f"Failed to create instance: {data}")
        logger.info("Created instance #%s from offer #%s", instance_id, offer_id)
        return int(instance_id)

    def wait_until_running(
        self,
        instance_id: int,
        timeout_s: int = 300,
        poll_interval: float = 5.0,
    ) -> VastInstance:
        """Poll until instance is running. Raises VastError on timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            info = self.get_instance(instance_id)
            if info.status == "running":
                return info
            logger.debug(
                "Instance #%s status: %s, waiting...", instance_id, info.status
            )
            time.sleep(poll_interval)
        raise VastError(f"Instance #{instance_id} did not start within {timeout_s}s")

    def get_instance(self, instance_id: int) -> VastInstance:
        """Get current instance info."""
        result = self._run(["show", "instance", str(instance_id), "--raw"])
        data = json.loads(result)

        ssh_host, ssh_port = self._parse_ssh_info(instance_id, data)

        return VastInstance(
            instance_id=instance_id,
            gpu_name=data.get("gpu_name", "unknown"),
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            price_per_hour=float(data.get("dph_total", 0)),
            status=data.get("actual_status", "unknown"),
        )

    def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance."""
        self._run(["destroy", "instance", str(instance_id)])
        logger.info("Destroyed instance #%s", instance_id)

    def get_ssh_url(self, instance_id: int) -> tuple[str, int]:
        """Return (host, port) for SSH connection."""
        result = self._run(["ssh-url", str(instance_id)])
        # Format: ssh://root@host:port
        url = result.strip()
        if "://" in url:
            url = url.split("://", 1)[1]
        if "@" in url:
            url = url.split("@", 1)[1]
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            return host, int(port_str)
        return url, 22

    # -- internals ---------------------------------------------------------

    def _parse_ssh_info(self, instance_id: int, data: dict) -> tuple[str, int]:
        """Extract SSH host and port from instance data."""
        import contextlib

        ssh_host = data.get("ssh_host", "")
        ssh_port = int(data.get("ssh_port", 22))
        if not ssh_host:
            with contextlib.suppress(Exception):
                ssh_host, ssh_port = self.get_ssh_url(instance_id)
        return ssh_host, ssh_port

    def _run(self, args: list[str]) -> str:
        """Run a vastai CLI command and return stdout."""
        cmd = ["vastai", *args]
        if self._api_key:
            cmd.extend(["--api-key", self._api_key])
        logger.debug("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
        except FileNotFoundError:
            raise VastError(
                "vastai CLI not found. Install with: pip install vastai"
            ) from None
        except subprocess.CalledProcessError as exc:
            raise VastError(
                f"vastai command failed: {' '.join(cmd)}\n{exc.stderr}"
            ) from exc
        return result.stdout
