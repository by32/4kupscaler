"""Tests for vast.ai client wrapper."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from upscaler.cloud.vast import VastClient, VastError


@pytest.fixture()
def client() -> VastClient:
    return VastClient(api_key="test-key")


class TestSearchOffers:
    def test_parses_offers(self, client):
        offers = [
            {"id": 123, "gpu_name": "RTX 3090", "dph_total": 0.25},
            {"id": 456, "gpu_name": "RTX 3090", "dph_total": 0.30},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps(offers), returncode=0)
            result = client.search_offers(gpu_filter="RTX_3090")
            assert len(result) == 2
            assert result[0]["id"] == 123

    def test_returns_empty_on_no_offers(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="[]", returncode=0)
            result = client.search_offers()
            assert result == []

    def test_includes_interruptible_flag(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="[]", returncode=0)
            client.search_offers(interruptible=True)
            args = mock_run.call_args[0][0]
            assert "--type" in args
            assert "interruptible" in args

    def test_max_price_filter(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="[]", returncode=0)
            client.search_offers(max_price_per_hour=0.50)
            args = mock_run.call_args[0][0]
            cmd_str = " ".join(args)
            assert "dph_total<=0.5" in cmd_str


class TestCreateInstance:
    def test_returns_instance_id(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=json.dumps({"new_contract": 789}), returncode=0
            )
            instance_id = client.create_instance(offer_id=123)
            assert instance_id == 789

    def test_raises_on_failure(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps({}), returncode=0)
            with pytest.raises(VastError, match="Failed to create"):
                client.create_instance(offer_id=123)


class TestDestroyInstance:
    def test_calls_destroy(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            client.destroy_instance(789)
            args = mock_run.call_args[0][0]
            assert "destroy" in args
            assert "789" in args


class TestGetSshUrl:
    def test_parses_ssh_url(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="ssh://root@host.example.com:12345\n", returncode=0
            )
            host, port = client.get_ssh_url(789)
            assert host == "host.example.com"
            assert port == 12345


class TestVastCliMissing:
    def test_raises_clear_error(self, client):
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            pytest.raises(VastError, match="vastai CLI not found"),
        ):
            client.search_offers()

    def test_raises_on_cli_error(self, client):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "vastai", stderr="some error"
            )
            with pytest.raises(VastError, match="command failed"):
                client.search_offers()
