from typing import Any

import numpy as np
import pytest

from formflow.pose import BodyPart, Keypoint, Pose
from formflow.pose_estimation import (
    PoseAnnotator,
    PoseEstimationError,
    YoloPoseEstimator,
)


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.array(values)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeKeypoints:
    def __init__(self, coordinates: Any, confidences: Any) -> None:
        self.xy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)


class FakeBoxes:
    def __init__(self, boxes: Any) -> None:
        self.xyxy = FakeTensor(boxes)
        self._length = len(boxes)

    def __len__(self) -> int:
        return self._length


class FakeResult:
    def __init__(self, coordinates: Any, confidences: Any, boxes: Any) -> None:
        self.keypoints = FakeKeypoints(coordinates, confidences)
        self.boxes = FakeBoxes(boxes)


class FakeModel:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.options: dict[str, Any] | None = None

    def predict(self, **options: Any) -> list[FakeResult]:
        self.options = options
        return self.results


def make_person(offset: float, confidence: float) -> tuple[list[list[float]], list[float]]:
    coordinates = [
        [offset + index, offset + index * 2]
        for index in range(len(BodyPart))
    ]
    confidences = [confidence] * len(BodyPart)
    return coordinates, confidences


def test_estimator_selects_person_with_largest_box() -> None:
    small_person, small_confidence = make_person(10, 0.7)
    large_person, large_confidence = make_person(100, 0.9)
    model = FakeModel(
        [
            FakeResult(
                [small_person, large_person],
                [small_confidence, large_confidence],
                [[0, 0, 20, 20], [0, 0, 80, 100]],
            )
        ]
    )
    estimator = YoloPoseEstimator(confidence=0.4, device="cpu", model=model)

    pose = estimator.estimate(np.zeros((120, 120, 3), dtype=np.uint8))

    assert pose is not None
    assert pose[BodyPart.NOSE].x == 100
    assert pose[BodyPart.NOSE].confidence == pytest.approx(0.9)
    assert model.options is not None
    assert model.options["conf"] == 0.4
    assert model.options["device"] == "cpu"


def test_estimator_returns_none_when_no_person_is_visible() -> None:
    model = FakeModel([])
    estimator = YoloPoseEstimator(model=model)

    pose = estimator.estimate(np.zeros((32, 32, 3), dtype=np.uint8))

    assert pose is None


def test_estimator_wraps_model_failure() -> None:
    class BrokenModel:
        def predict(self, **options: Any) -> None:
            raise RuntimeError("failure")

    estimator = YoloPoseEstimator(model=BrokenModel())

    with pytest.raises(PoseEstimationError, match="failed"):
        estimator.estimate(np.zeros((32, 32, 3), dtype=np.uint8))


def test_annotator_draws_visible_pose() -> None:
    class FixedEstimator:
        def estimate(self, frame: np.ndarray) -> Pose:
            points = [Keypoint(60, 60, 0.9) for _ in range(len(BodyPart))]
            points[BodyPart.LEFT_SHOULDER] = Keypoint(30, 50, 0.9)
            points[BodyPart.LEFT_ELBOW] = Keypoint(40, 70, 0.9)
            points[BodyPart.LEFT_WRIST] = Keypoint(50, 90, 0.9)
            return Pose(tuple(points))

    annotator = PoseAnnotator(FixedEstimator(), keypoint_confidence=0.5)
    frame = np.zeros((120, 300, 3), dtype=np.uint8)

    output = annotator(frame, frame_index=0)

    assert np.any(output != 0)
    assert annotator.frames_processed == 1
    assert annotator.frames_with_pose == 1


def test_annotator_handles_frame_without_pose() -> None:
    class EmptyEstimator:
        def estimate(self, frame: np.ndarray) -> None:
            return None

    annotator = PoseAnnotator(EmptyEstimator())

    output = annotator(np.zeros((80, 300, 3), dtype=np.uint8), frame_index=2)

    assert np.any(output != 0)
    assert annotator.frames_processed == 1
    assert annotator.frames_with_pose == 0
