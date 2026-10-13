"""Structured session summaries for FormFlow analysis runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepetitionSummary:
    index: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    minimum_joint_angle: float


@dataclass(frozen=True)
class SessionSummary:
    exercise: str
    frames_processed: int
    frames_with_measurement: int
    measurement_coverage: float
    processing_seconds: float
    processing_fps: float
    completed_reps: int
    incomplete_attempts: int
    repetitions: tuple[RepetitionSummary, ...]

    def __post_init__(self) -> None:
        if self.frames_processed <= 0:
            raise ValueError("frames_processed must be greater than zero.")
        if not 0 <= self.frames_with_measurement <= self.frames_processed:
            raise ValueError(
                "frames_with_measurement must be between zero and frames_processed."
            )
        if not 0.0 <= self.measurement_coverage <= 1.0:
            raise ValueError("measurement_coverage must be between 0 and 1.")
        if self.completed_reps != len(self.repetitions):
            raise ValueError("completed_reps must match the repetition list.")

    def write_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def measurement_coverage(measured_frames: int, total_frames: int) -> float:
    if total_frames <= 0:
        raise ValueError("total_frames must be greater than zero.")
    if not 0 <= measured_frames <= total_frames:
        raise ValueError("measured_frames must be between zero and total_frames.")
    return measured_frames / total_frames
