from pathlib import Path

import cv2
import numpy as np
import pytest

from formflow.video import VideoProcessingError, inspect_video, process_video


def write_sample_video(path: Path, *, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_inspect_video_returns_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample.avi"
    write_sample_video(source)

    info = inspect_video(source)

    assert info.width == 64
    assert info.height == 48
    assert info.fps == pytest.approx(12.0)
    assert info.frame_count == 6


def test_process_video_applies_transform_and_limits_frames(tmp_path: Path) -> None:
    source = tmp_path / "sample.avi"
    destination = tmp_path / "nested" / "output.avi"
    write_sample_video(source)
    visited_indices: list[int] = []

    def mark_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
        visited_indices.append(frame_index)
        frame[0:8, 0:8] = (0, 255, 0)
        return frame

    stats = process_video(
        source,
        destination,
        transform=mark_frame,
        max_frames=4,
        codec="MJPG",
    )

    assert destination.is_file()
    assert stats.frames_processed == 4
    assert stats.processing_fps > 0
    assert visited_indices == [0, 1, 2, 3]
    assert inspect_video(destination).frame_count == 4


def test_process_video_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(VideoProcessingError, match="does not exist"):
        process_video(tmp_path / "missing.mp4", tmp_path / "output.mp4")


def test_process_video_rejects_changed_frame_shape(tmp_path: Path) -> None:
    source = tmp_path / "sample.avi"
    write_sample_video(source, frame_count=1)

    with pytest.raises(VideoProcessingError, match="changed the frame"):
        process_video(
            source,
            tmp_path / "output.avi",
            transform=lambda frame, _: frame[:20, :20],
            codec="MJPG",
        )


def test_process_video_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        process_video(
            tmp_path / "unused.mp4",
            tmp_path / "output.mp4",
            max_frames=0,
        )
