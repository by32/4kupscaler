"""Tests for config schema validation."""

from pathlib import Path

import pytest

from upscaler.config.schema import BlockSwapConfig, UpscaleConfig


class TestBlockSwapConfig:
    def test_defaults(self) -> None:
        config = BlockSwapConfig()
        assert config.blocks_to_swap == 20
        assert config.use_non_blocking is True
        assert config.offload_io_components is False

    def test_rejects_negative_blocks(self) -> None:
        with pytest.raises(ValueError):
            BlockSwapConfig(blocks_to_swap=-1)

    def test_rejects_blocks_over_max(self) -> None:
        with pytest.raises(ValueError):
            BlockSwapConfig(blocks_to_swap=37)


class TestUpscaleConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video)
        assert config.model == "3b-fp8"
        assert config.batch_size == 1
        assert config.preserve_vram is True
        assert config.resolution == 1072
        assert config.output_format == "video"

    def test_auto_output_path(self, tmp_path: Path) -> None:
        video = tmp_path / "movie.mp4"
        video.touch()
        config = UpscaleConfig(input=video)
        assert config.output == tmp_path / "movie_upscaled.mp4"

    def test_valid_batch_sizes(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        for size in [1, 5, 9, 13, 17, 21]:
            config = UpscaleConfig(input=video, batch_size=size)
            assert config.batch_size == size

    def test_invalid_batch_size(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        for size in [2, 3, 4, 6, 7, 8, 10]:
            with pytest.raises(ValueError, match="4n\\+1"):
                UpscaleConfig(input=video, batch_size=size)

    def test_3b_model_max_blocks(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        with pytest.raises(ValueError, match="max is 32"):
            UpscaleConfig(
                input=video,
                model="3b-fp8",
                block_swap=BlockSwapConfig(blocks_to_swap=33),
            )

    def test_7b_model_allows_more_blocks(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(
            input=video,
            model="7b-fp8",
            block_swap=BlockSwapConfig(blocks_to_swap=35),
        )
        assert config.block_swap.blocks_to_swap == 35

    def test_segment_size_default_none(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        config = UpscaleConfig(input=video)
        assert config.segment_size is None

    def test_valid_segment_sizes(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        for size in [1, 5, 9, 13, 17, 21]:
            config = UpscaleConfig(input=video, segment_size=size)
            assert config.segment_size == size

    def test_invalid_segment_size(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        for size in [2, 3, 4, 6, 7, 8, 10]:
            with pytest.raises(ValueError, match="4n\\+1"):
                UpscaleConfig(input=video, segment_size=size)

    def test_invalid_output_format(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        with pytest.raises(ValueError):
            UpscaleConfig(input=video, output_format="gif")
