"""Tests for config schema validation."""

from pathlib import Path

import pytest

from upscaler.config.schema import BlockSwapConfig, UpscaleConfig, VAETilingConfig


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


class TestVAETilingConfig:
    def test_defaults(self) -> None:
        cfg = VAETilingConfig()
        assert cfg.encode_tiled is False
        assert cfg.decode_tiled is False
        assert cfg.encode_tile_size == 512
        assert cfg.encode_tile_overlap == 160
        assert cfg.decode_tile_size == 512
        assert cfg.decode_tile_overlap == 160

    def test_encode_overlap_must_be_less_than_tile_size(self) -> None:
        with pytest.raises(ValueError, match="encode_tile_overlap"):
            VAETilingConfig(
                encode_tiled=True,
                encode_tile_size=256,
                encode_tile_overlap=256,
            )

    def test_decode_overlap_must_be_less_than_tile_size(self) -> None:
        with pytest.raises(ValueError, match="decode_tile_overlap"):
            VAETilingConfig(
                decode_tiled=True,
                decode_tile_size=256,
                decode_tile_overlap=300,
            )

    def test_overlap_validation_skipped_when_tiling_disabled(self) -> None:
        cfg = VAETilingConfig(
            encode_tiled=False,
            encode_tile_size=256,
            encode_tile_overlap=300,
        )
        assert cfg.encode_tile_overlap == 300


class TestAutoEnableTiling:
    def test_auto_enabled_at_2160(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        cfg = UpscaleConfig(input=video, resolution=2160)
        assert cfg.vae_tiling.encode_tiled is True
        assert cfg.vae_tiling.decode_tiled is True

    def test_auto_enabled_at_1440(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        cfg = UpscaleConfig(input=video, resolution=1440)
        assert cfg.vae_tiling.encode_tiled is True
        assert cfg.vae_tiling.decode_tiled is True

    def test_not_auto_enabled_at_1080(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        cfg = UpscaleConfig(input=video, resolution=1080)
        assert cfg.vae_tiling.encode_tiled is False
        assert cfg.vae_tiling.decode_tiled is False

    def test_not_auto_enabled_at_1072(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        cfg = UpscaleConfig(input=video, resolution=1072)
        assert cfg.vae_tiling.encode_tiled is False
        assert cfg.vae_tiling.decode_tiled is False

    def test_explicit_tiling_preserved_at_low_res(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.touch()
        cfg = UpscaleConfig(
            input=video,
            resolution=720,
            vae_tiling=VAETilingConfig(encode_tiled=True, decode_tiled=True),
        )
        assert cfg.vae_tiling.encode_tiled is True
        assert cfg.vae_tiling.decode_tiled is True


class TestPresetVAETiling:
    def test_rtx3080_4k_has_tiling(self) -> None:
        from upscaler.core.presets import get_preset

        preset = get_preset("rtx3080-4k")
        assert preset["vae_tiling"]["encode_tiled"] is True
        assert preset["vae_tiling"]["decode_tiled"] is True
        assert preset["resolution"] == 2160

    def test_rtx3080_no_tiling(self) -> None:
        from upscaler.core.presets import get_preset

        preset = get_preset("rtx3080")
        assert "vae_tiling" not in preset
        assert preset["resolution"] == 1072
