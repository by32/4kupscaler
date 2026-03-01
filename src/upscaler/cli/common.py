"""Shared CLI options, validators, and callbacks."""

SUPPORTED_FORMATS = {".mp4", ".mpg", ".mpeg", ".avi", ".mkv", ".mov"}


def validate_batch_size(value: int) -> int:
    """Validate that batch_size follows the 4n+1 rule."""
    if (value - 1) % 4 != 0:
        valid = [4 * n + 1 for n in range(6)]
        msg = f"batch_size must follow 4n+1 rule (e.g., {valid}), got {value}"
        raise ValueError(msg)
    return value
