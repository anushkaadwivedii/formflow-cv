from math import cos, radians, sin

import numpy as np
import pytest

from formflow.curl import CurlAnalyzer, CurlPhase, CurlVideoAnnotator
from formflow.pose import BodyPart, BodySide, Keypoint, Pose


def curl_pose(elbow_angle: float, *, confidence: float = 0.95) -> Pose:
    points = [Keypoint(0, 0, 0.1) for _ in range(len(BodyPart))]
    wrist_x = sin(radians(elbow_angle)) * 100
    wrist_y = -cos(radians(elbow_angle)) * 100

    points[BodyPart.LEFT_SHOULDER] = Keypoint(100, 50, confidence)
    points[BodyPart.LEFT_ELBOW] = Keypoint(100, 150, confidence)
    points[BodyPart.LEFT_WRIST] = Keypoint(
        100 + wrist_x,
        150 + wrist_y,
        confidence,
    )
    return Pose(tuple(points))


def test_analyzer_counts_complete_curl() -> None:
    analyzer = CurlAnalyzer(smoothing=1.0)
    sequence = [165, 140, 100, 50, 45, 70, 110, 160]

    results = [
        analyzer.update(curl_pose(angle), frame_index)
        for frame_index, angle in enumerate(sequence)
    ]

    assert results[-1].completed_reps == 1
    assert results[-1].phase is CurlPhase.EXTENDED
    assert analyzer.completed[0].minimum_elbow_angle == pytest.approx(45)
    assert analyzer.completed[0].duration_frames == 7


def test_analyzer_records_incomplete_attempt() -> None:
    analyzer = CurlAnalyzer(smoothing=1.0)
    sequence = [165, 140, 85, 120, 160]

    result = None
    for frame_index, angle in enumerate(sequence):
        result = analyzer.update(curl_pose(angle), frame_index)

    assert result is not None
    assert result.completed_reps == 0
    assert result.incomplete_attempts == 1
    assert result.phase is CurlPhase.EXTENDED


def test_analyzer_ignores_pose_with_low_confidence() -> None:
    analyzer = CurlAnalyzer(smoothing=1.0)

    result = analyzer.update(curl_pose(50, confidence=0.2), frame_index=0)

    assert result.measurement_valid is False
    assert result.elbow_angle is None
    assert result.phase is CurlPhase.UNKNOWN


def test_analyzer_uses_visible_body_side() -> None:
    analyzer = CurlAnalyzer(smoothing=1.0)

    result = analyzer.update(curl_pose(165), frame_index=0)

    assert result.measurement_valid is True
    assert result.side is BodySide.LEFT
    assert result.elbow_angle == pytest.approx(165)


def test_analyzer_smooths_angle_changes() -> None:
    analyzer = CurlAnalyzer(smoothing=0.5)

    first = analyzer.update(curl_pose(170), frame_index=0)
    second = analyzer.update(curl_pose(130), frame_index=1)

    assert first.elbow_angle == pytest.approx(170)
    assert second.elbow_angle == pytest.approx(150)


def test_video_annotator_draws_curl_status() -> None:
    class FixedEstimator:
        def estimate(self, frame: np.ndarray) -> Pose:
            return curl_pose(165)

    annotator = CurlVideoAnnotator(
        FixedEstimator(),
        analyzer=CurlAnalyzer(smoothing=1.0),
    )

    output = annotator(np.zeros((300, 400, 3), dtype=np.uint8), frame_index=0)

    assert np.any(output != 0)
    assert annotator.frames_processed == 1
    assert annotator.frames_with_measurement == 1
