"""Pydantic models for configuration validation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class BlockSwapConfig(BaseModel):
    """BlockSwap memory optimization settings."""

    blocks_to_swap: int = Field(
        default=20,
        ge=0,
        le=36,
        description="Number of transformer blocks to offload to CPU (0=disabled).",
    )
    use_non_blocking: bool = True
    offload_io_components: bool = False
    cache_model: bool = False


class UpscaleConfig(BaseModel):
    """Full configuration for an upscale job."""

    # Input/Output
    input: Path
    output: Path | None = None
    output_format: str = Field(
        default="video",
        pattern=r"^(video|png)$",
        description="Output as video file or individual PNG frames.",
    )

    # Model
    model: str = Field(
        default="3b-fp8",
        description="Model variant: 3b-fp8, 3b-fp16, 7b-fp8, 7b-fp16, etc.",
    )
    model_dir: Path = Field(
        default=Path("~/.cache/upscaler/models"),
        description="Directory for downloaded model weights.",
    )

    # Resolution
    resolution: int = Field(
        default=1072,
        gt=0,
        description="Target short-side resolution in pixels.",
    )

    # Performance
    batch_size: int = Field(
        default=1,
        gt=0,
        description="Frames per batch (must follow 4n+1 rule: 1, 5, 9, 13, ...).",
    )
    seed: int = Field(default=100)
    preserve_vram: bool = Field(
        default=True,
        description="Unload unused models during processing to save VRAM.",
    )
    block_swap: BlockSwapConfig = Field(default_factory=BlockSwapConfig)
    cuda_device: str = Field(default="0")

    # Video processing
    skip_first_frames: int = Field(default=0, ge=0)
    max_frames: int | None = Field(default=None, gt=0)

    # Segmented processing
    segment_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Frames per segment for streaming processing"
            " (must follow 4n+1 rule). None = single pass."
        ),
    )

    # Color
    color_correction: bool = False

    @model_validator(mode="after")
    def validate_batch_size_rule(self) -> UpscaleConfig:
        """Ensure batch_size follows the 4n+1 pattern."""
        if (self.batch_size - 1) % 4 != 0:
            valid = [4 * n + 1 for n in range(6)]
            msg = (
                f"batch_size must follow 4n+1 rule "
                f"(valid: {valid}), got {self.batch_size}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_segment_size_rule(self) -> UpscaleConfig:
        """Ensure segment_size follows the 4n+1 pattern when set."""
        if self.segment_size is not None and (self.segment_size - 1) % 4 != 0:
            valid = [4 * n + 1 for n in range(6)]
            msg = (
                f"segment_size must follow 4n+1 rule "
                f"(valid: {valid}), got {self.segment_size}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_blocks_for_model(self) -> UpscaleConfig:
        """Ensure blocks_to_swap is within range for the selected model."""
        max_blocks = 32 if self.model.startswith("3b") else 36
        if self.block_swap.blocks_to_swap > max_blocks:
            msg = (
                f"blocks_to_swap max is {max_blocks} for {self.model} model, "
                f"got {self.block_swap.blocks_to_swap}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def set_default_output(self) -> UpscaleConfig:
        """Set output path based on input if not specified."""
        if self.output is None:
            stem = self.input.stem
            suffix = ".mp4" if self.output_format == "video" else ""
            if suffix:
                self.output = self.input.parent / f"{stem}_upscaled{suffix}"
            else:
                self.output = self.input.parent / f"{stem}_upscaled"
        return self
