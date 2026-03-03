"""Tests for video I/O utilities."""

import pytest

from upscaler.core.video_io import VideoMeta, compute_segments, enforce_4n1


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


class TestComputeSegments:
    def test_exact_division(self) -> None:
        segments = compute_segments(10, 5)
        assert segments == [(0, 5), (5, 10)]

    def test_remainder(self) -> None:
        segments = compute_segments(7, 5)
        assert segments == [(0, 5), (5, 7)]

    def test_fewer_than_segment_size(self) -> None:
        segments = compute_segments(3, 5)
        assert segments == [(0, 3)]

    def test_equal_to_segment_size(self) -> None:
        segments = compute_segments(5, 5)
        assert segments == [(0, 5)]

    def test_zero_frames(self) -> None:
        segments = compute_segments(0, 5)
        assert segments == []

    def test_single_frame(self) -> None:
        segments = compute_segments(1, 5)
        assert segments == [(0, 1)]

    def test_large_segment_count(self) -> None:
        segments = compute_segments(25, 5)
        assert len(segments) == 5
        # All frames covered
        assert segments[0][0] == 0
        assert segments[-1][1] == 25

    def test_segments_cover_all_frames(self) -> None:
        for total in range(1, 50):
            for seg_size in [1, 5, 9, 13]:
                segments = compute_segments(total, seg_size)
                assert segments[0][0] == 0
                assert segments[-1][1] == total
                # No gaps
                for i in range(1, len(segments)):
                    assert segments[i][0] == segments[i - 1][1]


class TestReadVideoSegment:
    @pytest.fixture()
    def sample_video(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 48))
        for i in range(10):
            frame = np.full((48, 64, 3), i * 25, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return video_path

    def test_reads_correct_range(self, sample_video) -> None:
        pytest.importorskip("torch")
        from upscaler.core.video_io import read_video_segment

        tensor, meta = read_video_segment(sample_video, 2, 5)
        assert tensor.shape[0] == 3  # 3 frames: index 2,3,4

    def test_reads_full_video(self, sample_video) -> None:
        pytest.importorskip("torch")
        from upscaler.core.video_io import read_video_segment

        tensor, meta = read_video_segment(sample_video, 0, 10)
        assert tensor.shape[0] == 10

    def test_reads_first_segment(self, sample_video) -> None:
        pytest.importorskip("torch")
        from upscaler.core.video_io import read_video_segment

        tensor, _ = read_video_segment(sample_video, 0, 5)
        assert tensor.shape[0] == 5

    def test_meta_preserved(self, sample_video) -> None:
        pytest.importorskip("torch")
        from upscaler.core.video_io import read_video_segment

        _, meta = read_video_segment(sample_video, 0, 5)
        assert meta.width == 64
        assert meta.height == 48
        assert meta.fps == 30.0


class TestStreamingVideoWriter:
    def test_writes_across_segments(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        torch = pytest.importorskip("torch")
        from upscaler.core.video_io import StreamingVideoWriter

        out = tmp_path / "out.mp4"
        with StreamingVideoWriter(out, fps=30.0, width=64, height=48) as writer:
            seg1 = torch.rand(3, 48, 64, 3)
            writer.write_tensor(seg1)
            seg2 = torch.rand(2, 48, 64, 3)
            writer.write_tensor(seg2)
            assert writer.frame_count == 5

        # Verify the output file exists and has the right frame count
        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert count == 5


class TestStreamingPngWriter:
    def test_writes_across_segments(self, tmp_path) -> None:
        pytest.importorskip("cv2")
        torch = pytest.importorskip("torch")
        from upscaler.core.video_io import StreamingPngWriter

        out_dir = tmp_path / "pngs"
        with StreamingPngWriter(out_dir) as writer:
            seg1 = torch.rand(3, 48, 64, 3)
            writer.write_tensor(seg1)
            seg2 = torch.rand(2, 48, 64, 3)
            writer.write_tensor(seg2)
            assert writer.frame_count == 5

        pngs = sorted(out_dir.glob("*.png"))
        assert len(pngs) == 5
        # Verify sequential naming
        assert pngs[0].name == "frame_000000.png"
        assert pngs[4].name == "frame_000004.png"


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
