"""Video input and output utilities for the FormFlow pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray


Frame = NDArray[np.uint8]
FrameTransform = Callable[[Frame, int], Frame]


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be read, transformed, or written safely."""


@dataclass(frozen=True)
class VideoInfo:
    """Properties needed to process a video without changing its geometry."""

    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class ProcessingStats:
    """Measured results from a completed video processing run."""

    input_path: Path
    output_path: Path
    frames_processed: int
    source_fps: float
    elapsed_seconds: float
    processing_fps: float


def _read_video_info(capture: cv2.VideoCapture) -> VideoInfo:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise VideoProcessingError("Video has invalid frame dimensions.")
    if fps <= 0:
        raise VideoProcessingError("Video does not report a valid frame rate.")

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=max(frame_count, 0),
    )


def inspect_video(input_path: str | Path) -> VideoInfo:
    """Open a video and return validated metadata."""

    path = Path(input_path)
    if not path.is_file():
        raise VideoProcessingError(f"Input video does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoProcessingError(f"Could not open input video: {path}")
        return _read_video_info(capture)
    finally:
        capture.release()


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    transform: FrameTransform | None = None,
    max_frames: int | None = None,
    codec: str = "mp4v",
) -> ProcessingStats:
    """Process a video frame by frame and write an annotated-ready output.

    The transform receives a copy of each frame and its zero-based index. It
    must return a color frame with the same dimensions as the input.
    """

    source = Path(input_path)
    destination = Path(output_path)

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters.")
    if not source.is_file():
        raise VideoProcessingError(f"Input video does not exist: {source}")

    capture = cv2.VideoCapture(str(source))
    writer: cv2.VideoWriter | None = None
    frames_processed = 0
    started_at = perf_counter()

    try:
        if not capture.isOpened():
            raise VideoProcessingError(f"Could not open input video: {source}")

        info = _read_video_info(capture)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(destination),
            fourcc,
            info.fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise VideoProcessingError(f"Could not open output video: {destination}")

        while max_frames is None or frames_processed < max_frames:
            success, frame = capture.read()
            if not success:
                break

            output_frame = (
                transform(frame.copy(), frames_processed)
                if transform is not None
                else frame
            )
            if not isinstance(output_frame, np.ndarray):
                raise VideoProcessingError("Frame transform must return a NumPy array.")
            if output_frame.shape != frame.shape:
                raise VideoProcessingError(
                    "Frame transform changed the frame dimensions or channels."
                )

            writer.write(output_frame)
            frames_processed += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_seconds = perf_counter() - started_at
    if frames_processed == 0:
        raise VideoProcessingError("The input video contained no readable frames.")

    return ProcessingStats(
        input_path=source,
        output_path=destination,
        frames_processed=frames_processed,
        source_fps=info.fps,
        elapsed_seconds=elapsed_seconds,
        processing_fps=frames_processed / elapsed_seconds,
    )
