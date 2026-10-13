"""Pose keypoints and geometry shared by FormFlow exercise analyzers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import acos, degrees, hypot
from typing import Iterable


class BodyPart(IntEnum):
    """COCO body keypoint indices returned by the pose model."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class BodySide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Keypoint:
    """One model-estimated body landmark in pixel coordinates."""

    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Keypoint confidence must be between 0 and 1.")

    def is_visible(self, minimum_confidence: float = 0.5) -> bool:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")
        return self.confidence >= minimum_confidence


@dataclass(frozen=True)
class Pose:
    """The seventeen COCO body keypoints estimated for one person."""

    keypoints: tuple[Keypoint, ...]

    def __post_init__(self) -> None:
        if len(self.keypoints) != len(BodyPart):
            raise ValueError(f"Pose must contain exactly {len(BodyPart)} keypoints.")

    def __getitem__(self, body_part: BodyPart) -> Keypoint:
        return self.keypoints[int(body_part)]

    def all_visible(
        self,
        body_parts: Iterable[BodyPart],
        *,
        minimum_confidence: float = 0.5,
    ) -> bool:
        return all(
            self[body_part].is_visible(minimum_confidence)
            for body_part in body_parts
        )


def joint_angle(first: Keypoint, vertex: Keypoint, third: Keypoint) -> float:
    """Return the smaller angle at ``vertex`` in degrees from 0 to 180."""

    first_vector = (first.x - vertex.x, first.y - vertex.y)
    third_vector = (third.x - vertex.x, third.y - vertex.y)
    first_length = hypot(*first_vector)
    third_length = hypot(*third_vector)

    if first_length == 0 or third_length == 0:
        raise ValueError("Cannot calculate an angle from overlapping keypoints.")

    dot_product = (
        first_vector[0] * third_vector[0]
        + first_vector[1] * third_vector[1]
    )
    cosine = dot_product / (first_length * third_length)
    cosine = min(1.0, max(-1.0, cosine))
    return degrees(acos(cosine))


def side_confidence(pose: Pose, body_parts: Iterable[BodyPart]) -> float:
    """Return the mean confidence of the requested landmarks."""

    points = [pose[body_part] for body_part in body_parts]
    if not points:
        raise ValueError("At least one body part is required.")
    return sum(point.confidence for point in points) / len(points)


def choose_body_side(
    pose: Pose,
    *,
    left_parts: Iterable[BodyPart],
    right_parts: Iterable[BodyPart],
) -> BodySide:
    """Choose the side whose required keypoints have higher mean confidence."""

    left_score = side_confidence(pose, left_parts)
    right_score = side_confidence(pose, right_parts)
    return BodySide.LEFT if left_score >= right_score else BodySide.RIGHT
