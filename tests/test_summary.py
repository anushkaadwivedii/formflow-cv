import json
from pathlib import Path

import pytest

from formflow.summary import (
    RepetitionSummary,
    SessionSummary,
    measurement_coverage,
)


def test_measurement_coverage_calculates_ratio() -> None:
    assert measurement_coverage(75, 100) == pytest.approx(0.75)


def test_measurement_coverage_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="between zero"):
        measurement_coverage(11, 10)


def test_session_summary_writes_json(tmp_path: Path) -> None:
    repetition = RepetitionSummary(
        index=1,
        start_frame=10,
        end_frame=39,
        duration_frames=30,
        duration_seconds=1.0,
        minimum_joint_angle=87.5,
    )
    summary = SessionSummary(
        exercise="squat",
        frames_processed=100,
        frames_with_measurement=90,
        measurement_coverage=0.9,
        processing_seconds=4.0,
        processing_fps=25.0,
        completed_reps=1,
        incomplete_attempts=0,
        repetitions=(repetition,),
    )

    output = summary.write_json(tmp_path / "nested" / "summary.json")
    contents = json.loads(output.read_text(encoding="utf-8"))

    assert contents["exercise"] == "squat"
    assert contents["completed_reps"] == 1
    assert contents["repetitions"][0]["minimum_joint_angle"] == 87.5


def test_session_summary_validates_repetition_count() -> None:
    with pytest.raises(ValueError, match="match"):
        SessionSummary(
            exercise="curl",
            frames_processed=10,
            frames_with_measurement=10,
            measurement_coverage=1.0,
            processing_seconds=1.0,
            processing_fps=10.0,
            completed_reps=1,
            incomplete_attempts=0,
            repetitions=(),
        )
