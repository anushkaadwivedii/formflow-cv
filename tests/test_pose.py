import pytest

from formflow.pose import (
    BodyPart,
    BodySide,
    Keypoint,
    Pose,
    choose_body_side,
    joint_angle,
)


def make_pose(default_confidence: float = 0.5) -> Pose:
    return Pose(
        tuple(
            Keypoint(float(index), float(index), default_confidence)
            for index in range(len(BodyPart))
        )
    )


def test_joint_angle_returns_right_angle() -> None:
    angle = joint_angle(
        Keypoint(1, 0, 1),
        Keypoint(0, 0, 1),
        Keypoint(0, 1, 1),
    )

    assert angle == pytest.approx(90.0)


def test_joint_angle_returns_straight_angle() -> None:
    angle = joint_angle(
        Keypoint(-1, 0, 1),
        Keypoint(0, 0, 1),
        Keypoint(1, 0, 1),
    )

    assert angle == pytest.approx(180.0)


def test_joint_angle_rejects_overlapping_points() -> None:
    point = Keypoint(0, 0, 1)

    with pytest.raises(ValueError, match="overlapping"):
        joint_angle(point, point, Keypoint(1, 0, 1))


def test_pose_requires_all_seventeen_keypoints() -> None:
    with pytest.raises(ValueError, match="exactly 17"):
        Pose((Keypoint(0, 0, 1),))


def test_pose_checks_keypoint_visibility() -> None:
    pose = make_pose(default_confidence=0.8)

    assert pose.all_visible(
        [BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST],
        minimum_confidence=0.7,
    )


def test_choose_body_side_uses_more_confident_landmarks() -> None:
    points = list(make_pose().keypoints)
    points[BodyPart.LEFT_HIP] = Keypoint(10, 10, 0.9)
    points[BodyPart.LEFT_KNEE] = Keypoint(10, 20, 0.9)
    points[BodyPart.RIGHT_HIP] = Keypoint(20, 10, 0.3)
    points[BodyPart.RIGHT_KNEE] = Keypoint(20, 20, 0.3)
    pose = Pose(tuple(points))

    selected = choose_body_side(
        pose,
        left_parts=[BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE],
        right_parts=[BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE],
    )

    assert selected is BodySide.LEFT
