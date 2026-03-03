"""Video read/write and frame constraint enforcement.

Frames are handled as tensors in [T, H, W, C] format, float32, range [0, 1],
RGB color order — matching the SeedVR2 inference pipeline expectations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mpg", ".mpeg", ".avi", ".mkv", ".mov"}


@dataclass
class VideoMeta:
    """Metadata extracted from a video file."""

    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


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
    return frame_count + (5 - remainder) % 4


def get_video_meta(path: Path) -> VideoMeta:
    """Read video metadata without loading frames.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        ValueError: If the file can't be opened as a video.
    """
    import cv2

    if not path.exists():
        msg = f"Video not found: {path}"
        raise FileNotFoundError(msg)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        msg = f"Cannot open video: {path}"
        raise ValueError(msg)

    try:
        meta = VideoMeta(
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()

    return meta


def read_video(
    path: Path,
    skip_frames: int = 0,
    max_frames: int | None = None,
) -> tuple:
    """Read video frames into a tensor.

    Args:
        path: Path to video file.
        skip_frames: Number of frames to skip from the start.
        max_frames: Maximum frames to read (None = all).

    Returns:
        Tuple of (frames_tensor, video_meta).
        Tensor shape: [T, H, W, C], dtype float32, range [0, 1], RGB.

    Raises:
        FileNotFoundError: If video doesn't exist.
        ValueError: If video can't be opened or has no frames.
    """
    import cv2
    import numpy as np
    import torch

    meta = get_video_meta(path)
    cap = cv2.VideoCapture(str(path))

    try:
        # Skip frames
        if skip_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

        limit = max_frames if max_frames else meta.frame_count - skip_frames
        frames = []
        for _ in range(limit):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        msg = f"No frames read from: {path}"
        raise ValueError(msg)

    tensor = torch.from_numpy(np.stack(frames)).to(torch.float32)
    logger.info(
        "Read %d frames from %s (%dx%d @ %.1f fps)",
        len(frames),
        path,
        meta.width,
        meta.height,
        meta.fps,
    )
    return tensor, meta


def pad_to_4n1(frames_tensor) -> tuple:
    """Pad a frames tensor so frame count satisfies 4n+1.

    Pads by repeating the last frame. Returns the padded tensor
    and the number of padding frames added.

    Args:
        frames_tensor: [T, H, W, C] tensor.

    Returns:
        Tuple of (padded_tensor, num_padded).
    """
    import torch

    t = frames_tensor.shape[0]
    target = enforce_4n1(t)
    pad_count = target - t

    if pad_count == 0:
        return frames_tensor, 0

    last_frame = frames_tensor[-1:].expand(pad_count, -1, -1, -1)
    padded = torch.cat([frames_tensor, last_frame], dim=0)
    logger.info("Padded %d -> %d frames (added %d)", t, target, pad_count)
    return padded, pad_count


def write_video(
    frames_tensor,
    output_path: Path,
    fps: float = 30.0,
) -> None:
    """Write a frames tensor to an MP4 video file.

    Args:
        frames_tensor: [T, H, W, C] float32 [0, 1] RGB tensor.
        output_path: Output .mp4 path.
        fps: Frames per second.
    """
    import cv2
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _c = frames_tensor.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        msg = f"Cannot create video writer for: {output_path}"
        raise ValueError(msg)

    try:
        for i in range(t):
            frame_np = (frames_tensor[i].cpu().numpy() * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    logger.info("Wrote %d frames to %s", t, output_path)


def write_frames_as_png(
    frames_tensor,
    output_dir: Path,
    base_name: str = "frame",
) -> int:
    """Save frames as individual PNG files.

    Args:
        frames_tensor: [T, H, W, C] float32 [0, 1] RGB tensor.
        output_dir: Directory for PNG output.
        base_name: Filename prefix (e.g., "frame" -> "frame_000000.png").

    Returns:
        Number of frames saved.
    """
    import cv2
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    t = frames_tensor.shape[0]

    for idx in range(t):
        filename = f"{base_name}_{idx:06d}.png"
        filepath = output_dir / filename
        frame_np = (frames_tensor[idx].cpu().numpy() * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), frame_bgr)

    logger.info("Saved %d PNGs to %s", t, output_dir)
    return t
