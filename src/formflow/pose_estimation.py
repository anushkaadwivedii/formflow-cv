"""Pose model integration and video annotation for FormFlow."""

from __future__ import annotations

from typing import Any, Protocol

import cv2

from formflow.pose import BodyPart, Keypoint, Pose
from formflow.video import Frame


class PoseEstimationError(RuntimeError):
    """Raised when the pose model cannot load or process a frame."""


class PoseEstimator(Protocol):
    """Interface for models that estimate one primary pose per frame."""

    def estimate(self, frame: Frame) -> Pose | None:
        """Return the primary pose, or None when no person is visible."""


class YoloPoseEstimator:
    """Adapter that converts Ultralytics pose output into FormFlow keypoints."""

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        *,
        confidence: float = 0.35,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be greater than 0 and at most 1.")

        if model is None:
            try:
                from ultralytics import YOLO

                model = YOLO(model_name)
            except Exception as error:
                raise PoseEstimationError(
                    f"Could not load pose model {model_name!r}."
                ) from error

        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self._model = model

    def estimate(self, frame: Frame) -> Pose | None:
        options: dict[str, Any] = {
            "source": frame,
            "conf": self.confidence,
            "verbose": False,
        }
        if self.device is not None:
            options["device"] = self.device

        try:
            results = self._model.predict(**options)
        except Exception as error:
            raise PoseEstimationError(
                "The pose model failed while processing a video frame."
            ) from error

        if not results:
            return None

        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return None
        if len(result.boxes) == 0:
            return None

        coordinates = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf
        if confidences is None:
            raise PoseEstimationError("The pose model did not return keypoint confidence.")
        confidence_values = confidences.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        if len(coordinates) == 0:
            return None

        primary_index = max(
            range(len(boxes)),
            key=lambda index: _box_area(boxes[index]),
        )
        primary_coordinates = coordinates[primary_index]
        primary_confidences = confidence_values[primary_index]

        if len(primary_coordinates) != len(BodyPart):
            raise PoseEstimationError(
                f"Expected {len(BodyPart)} keypoints, received "
                f"{len(primary_coordinates)}."
            )

        return Pose(
            tuple(
                Keypoint(
                    x=float(point[0]),
                    y=float(point[1]),
                    confidence=float(score),
                )
                for point, score in zip(
                    primary_coordinates,
                    primary_confidences,
                    strict=True,
                )
            )
        )


def _box_area(box: Any) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


class PoseAnnotator:
    """Draw visible keypoints and skeleton connections on each frame."""

    KEYPOINT_COLOR = (46, 204, 113)
    SKELETON_COLOR = (52, 152, 219)
    PANEL_COLOR = (24, 31, 38)
    TEXT_COLOR = (255, 255, 255)

    SKELETON = (
        (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
        (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
        (BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
        (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
        (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
        (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
        (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
        (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
        (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
        (BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
        (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
        (BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE),
    )

    def __init__(
        self,
        estimator: PoseEstimator,
        *,
        keypoint_confidence: float = 0.5,
    ) -> None:
        if not 0.0 <= keypoint_confidence <= 1.0:
            raise ValueError("keypoint_confidence must be between 0 and 1.")
        self.estimator = estimator
        self.keypoint_confidence = keypoint_confidence
        self.frames_processed = 0
        self.frames_with_pose = 0

    def __call__(self, frame: Frame, frame_index: int) -> Frame:
        pose = self.estimator.estimate(frame)
        self.frames_processed += 1

        if pose is not None:
            self.frames_with_pose += 1
            draw_pose(
                frame,
                pose,
                keypoint_confidence=self.keypoint_confidence,
            )

        status = "Pose detected" if pose is not None else "No pose"
        panel_text = f"Frame {frame_index + 1} | {status}"
        cv2.rectangle(frame, (8, 8), (260, 38), self.PANEL_COLOR, -1)
        cv2.putText(
            frame,
            panel_text,
            (16, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        return frame



def draw_pose(
    frame: Frame,
    pose: Pose,
    *,
    keypoint_confidence: float = 0.5,
) -> None:
    """Draw visible pose connections and keypoints on a frame."""

    for first_part, second_part in PoseAnnotator.SKELETON:
        first = pose[first_part]
        second = pose[second_part]
        if not first.is_visible(keypoint_confidence):
            continue
        if not second.is_visible(keypoint_confidence):
            continue
        cv2.line(
            frame,
            (round(first.x), round(first.y)),
            (round(second.x), round(second.y)),
            PoseAnnotator.SKELETON_COLOR,
            2,
            cv2.LINE_AA,
        )

    for keypoint in pose.keypoints:
        if keypoint.is_visible(keypoint_confidence):
            cv2.circle(
                frame,
                (round(keypoint.x), round(keypoint.y)),
                4,
                PoseAnnotator.KEYPOINT_COLOR,
                -1,
                cv2.LINE_AA,
            )
