from math import cos, radians, sin

import numpy as np
import pytest

from formflow.pose import BodyPart, BodySide, Keypoint, Pose
from formflow.squat import SquatAnalyzer, SquatPhase, SquatVideoAnnotator


def squat_pose(knee_angle: float, *, confidence: float = 0.95) -> Pose:
    points = [Keypoint(0, 0, 0.1) for _ in range(len(BodyPart))]
    ankle_x = sin(radians(knee_angle)) * 100
    ankle_y = -cos(radians(knee_angle)) * 100

    points[BodyPart.LEFT_SHOULDER] = Keypoint(100, 50, confidence)
    points[BodyPart.LEFT_HIP] = Keypoint(100, 150, confidence)
    points[BodyPart.LEFT_KNEE] = Keypoint(100, 250, confidence)
    points[BodyPart.LEFT_ANKLE] = Keypoint(
        100 + ankle_x,
        250 + ankle_y,
        confidence,
    )
    return Pose(tuple(points))


def test_analyzer_counts_complete_squat() -> None:
    analyzer = SquatAnalyzer(smoothing=1.0)
    sequence = [175, 150, 120, 95, 90, 115, 145, 165]

    results = [
        analyzer.update(squat_pose(angle), frame_index)
        for frame_index, angle in enumerate(sequence)
    ]

    assert results[-1].completed_reps == 1
    assert results[-1].phase is SquatPhase.STANDING
    assert analyzer.completed[0].minimum_knee_angle == pytest.approx(90)
    assert analyzer.completed[0].duration_frames == 7


def test_analyzer_records_incomplete_attempt() -> None:
    analyzer = SquatAnalyzer(smoothing=1.0)
    sequence = [175, 150, 135, 145, 165]

    result = None
    for frame_index, angle in enumerate(sequence):
        result = analyzer.update(squat_pose(angle), frame_index)

    assert result is not None
    assert result.completed_reps == 0
    assert result.incomplete_attempts == 1
    assert result.phase is SquatPhase.STANDING


def test_analyzer_ignores_pose_with_low_confidence() -> None:
    analyzer = SquatAnalyzer(smoothing=1.0)

    result = analyzer.update(squat_pose(90, confidence=0.2), frame_index=0)

    assert result.measurement_valid is False
    assert result.knee_angle is None
    assert result.phase is SquatPhase.UNKNOWN


def test_analyzer_uses_visible_body_side() -> None:
    analyzer = SquatAnalyzer(smoothing=1.0)

    result = analyzer.update(squat_pose(175), frame_index=0)

    assert result.measurement_valid is True
    assert result.side is BodySide.LEFT
    assert result.knee_angle == pytest.approx(175)


def test_analyzer_smooths_angle_changes() -> None:
    analyzer = SquatAnalyzer(smoothing=0.5)

    first = analyzer.update(squat_pose(170), frame_index=0)
    second = analyzer.update(squat_pose(130), frame_index=1)

    assert first.knee_angle == pytest.approx(170)
    assert second.knee_angle == pytest.approx(150)


def test_video_annotator_draws_squat_status() -> None:
    class FixedEstimator:
        def estimate(self, frame: np.ndarray) -> Pose:
            return squat_pose(175)

    annotator = SquatVideoAnnotator(
        FixedEstimator(),
        analyzer=SquatAnalyzer(smoothing=1.0),
    )

    output = annotator(np.zeros((400, 400, 3), dtype=np.uint8), frame_index=0)

    assert np.any(output != 0)
    assert annotator.frames_processed == 1
    assert annotator.frames_with_measurement == 1
