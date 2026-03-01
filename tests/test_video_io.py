"""Tests for video I/O utilities."""

import pytest

from upscaler.core.video_io import enforce_4n1


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
