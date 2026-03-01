"""Tests for video I/O utilities."""

import pytest

from upscaler.core.video_io import VideoMeta, enforce_4n1


class TestEnforce4n1:
    @pytest.mark.parametrize(
        ("input_count", "expected"),
        [
            (1, 1),  # Already valid
            (5, 5),  # Already valid
            (9, 9),  # Already valid
            (2, 5),  # Pad 2 -> 5
            (3, 5),  # Pad 3 -> 5
            (4, 5),  # Pad 4 -> 5
            (6, 9),  # Pad 6 -> 9
            (7, 9),  # Pad 7 -> 9
            (8, 9),  # Pad 8 -> 9
            (10, 13),  # Pad 10 -> 13
            (0, 1),  # Edge case: 0 -> 1
        ],
    )
    def test_enforce_4n1(self, input_count: int, expected: int) -> None:
        assert enforce_4n1(input_count) == expected

    def test_result_always_satisfies_rule(self) -> None:
        for n in range(100):
            result = enforce_4n1(n)
            assert result >= n or n == 0
            assert (result - 1) % 4 == 0


class TestVideoMeta:
    def test_duration(self) -> None:
        meta = VideoMeta(fps=30.0, frame_count=150, width=1920, height=1080)
        assert meta.duration == 5.0

    def test_duration_zero_fps(self) -> None:
        meta = VideoMeta(fps=0.0, frame_count=100, width=640, height=480)
        assert meta.duration == 0.0

    def test_fields(self) -> None:
        meta = VideoMeta(fps=24.0, frame_count=240, width=3840, height=2160)
        assert meta.fps == 24.0
        assert meta.frame_count == 240
        assert meta.width == 3840
        assert meta.height == 2160
