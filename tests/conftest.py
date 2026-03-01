"""Shared test fixtures."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def _skip_gpu_tests(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("gpu") and not has_cuda():
        pytest.skip("requires CUDA GPU")
