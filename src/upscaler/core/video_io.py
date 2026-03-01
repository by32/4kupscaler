"""Video read/write and frame constraint enforcement."""

from __future__ import annotations


def enforce_4n1(frame_count: int) -> int:
    """Return the next valid frame count satisfying frames % 4 == 1.

    SeedVR2 requires batch sizes following the 4n+1 pattern.
    If frame_count already satisfies, return it unchanged.
    Otherwise, return the next valid count (pad with repeated last frame).
    """
    if frame_count <= 0:
        return 1
    remainder = frame_count % 4
    if remainder == 1:
        return frame_count
    # Need to pad up to the next 4n+1
    return frame_count + (5 - remainder) % 4
