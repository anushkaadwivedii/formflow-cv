"""Squat angle measurement, phase tracking, and video annotation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2

from formflow.pose import (
    BodyPart,
    BodySide,
    Pose,
    choose_body_side,
    joint_angle,
)
from formflow.pose_estimation import PoseEstimator, draw_pose
from formflow.video import Frame


class SquatPhase(str, Enum):
    UNKNOWN = "unknown"
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


@dataclass(frozen=True)
class CompletedSquat:
    index: int
    start_frame: int
    end_frame: int
    minimum_knee_angle: float

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class SquatFrameResult:
    frame_index: int
    phase: SquatPhase
    side: BodySide | None
    knee_angle: float | None
    hip_angle: float | None
    completed_reps: int
    incomplete_attempts: int
    measurement_valid: bool


class SquatAnalyzer:
    """Convert a sequence of poses into squat phases and repetition counts."""

    LEFT_PARTS = (
        BodyPart.LEFT_SHOULDER,
        BodyPart.LEFT_HIP,
        BodyPart.LEFT_KNEE,
        BodyPart.LEFT_ANKLE,
    )
    RIGHT_PARTS = (
        BodyPart.RIGHT_SHOULDER,
        BodyPart.RIGHT_HIP,
        BodyPart.RIGHT_KNEE,
        BodyPart.RIGHT_ANKLE,
    )

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.5,
        smoothing: float = 0.35,
        standing_angle: float = 160.0,
        bottom_angle: float = 100.0,
        incomplete_angle: float = 140.0,
        hysteresis: float = 5.0,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be greater than 0 and at most 1.")
        if not 0.0 < bottom_angle < incomplete_angle < standing_angle < 180.0:
            raise ValueError(
                "Expected bottom_angle < incomplete_angle < standing_angle."
            )
        if hysteresis < 0:
            raise ValueError("hysteresis cannot be negative.")

        self.minimum_confidence = minimum_confidence
        self.smoothing = smoothing
        self.standing_angle = standing_angle
        self.bottom_angle = bottom_angle
        self.incomplete_angle = incomplete_angle
        self.hysteresis = hysteresis

        self.phase = SquatPhase.UNKNOWN
        self.completed: list[CompletedSquat] = []
        self.incomplete_attempts = 0
        self._smoothed_knee_angle: float | None = None
        self._attempt_start_frame: int | None = None
        self._attempt_minimum_angle: float | None = None

    def update(self, pose: Pose | None, frame_index: int) -> SquatFrameResult:
        measurement = self._measure(pose)
        if measurement is None:
            return self._result(frame_index, None, None, None, False)

        side, raw_knee_angle, hip_angle = measurement
        knee_angle = self._smooth(raw_knee_angle)
        self._update_phase(knee_angle, frame_index)
        return self._result(
            frame_index,
            side,
            knee_angle,
            hip_angle,
            True,
        )

    def _measure(self, pose: Pose | None) -> tuple[BodySide, float, float] | None:
        if pose is None:
            return None

        preferred_side = choose_body_side(
            pose,
            left_parts=self.LEFT_PARTS,
            right_parts=self.RIGHT_PARTS,
        )
        choices = (
            (preferred_side, self.LEFT_PARTS)
            if preferred_side is BodySide.LEFT
            else (preferred_side, self.RIGHT_PARTS)
        )
        fallback = (
            (BodySide.RIGHT, self.RIGHT_PARTS)
            if preferred_side is BodySide.LEFT
            else (BodySide.LEFT, self.LEFT_PARTS)
        )

        for side, parts in (choices, fallback):
            if not pose.all_visible(
                parts,
                minimum_confidence=self.minimum_confidence,
            ):
                continue
            shoulder, hip, knee, ankle = (pose[part] for part in parts)
            return (
                side,
                joint_angle(hip, knee, ankle),
                joint_angle(shoulder, hip, knee),
            )
        return None

    def _smooth(self, knee_angle: float) -> float:
        if self._smoothed_knee_angle is None:
            self._smoothed_knee_angle = knee_angle
        else:
            self._smoothed_knee_angle = (
                self.smoothing * knee_angle
                + (1.0 - self.smoothing) * self._smoothed_knee_angle
            )
        return self._smoothed_knee_angle

    def _update_phase(self, knee_angle: float, frame_index: int) -> None:
        if self.phase is SquatPhase.UNKNOWN:
            if knee_angle >= self.standing_angle:
                self.phase = SquatPhase.STANDING
            return

        if self.phase is SquatPhase.STANDING:
            if knee_angle < self.standing_angle - self.hysteresis:
                self.phase = SquatPhase.DESCENDING
                self._attempt_start_frame = frame_index
                self._attempt_minimum_angle = knee_angle
            return

        if self._attempt_minimum_angle is None:
            self._attempt_minimum_angle = knee_angle
        else:
            self._attempt_minimum_angle = min(
                self._attempt_minimum_angle,
                knee_angle,
            )

        if self.phase is SquatPhase.DESCENDING:
            if knee_angle <= self.bottom_angle:
                self.phase = SquatPhase.BOTTOM
            elif knee_angle >= self.standing_angle:
                if self._attempt_minimum_angle <= self.incomplete_angle:
                    self.incomplete_attempts += 1
                self._reset_attempt()
                self.phase = SquatPhase.STANDING
            return

        if self.phase is SquatPhase.BOTTOM:
            if knee_angle > self.bottom_angle + self.hysteresis:
                self.phase = SquatPhase.ASCENDING
            return

        if self.phase is SquatPhase.ASCENDING and knee_angle >= self.standing_angle:
            start_frame = self._attempt_start_frame
            minimum_angle = self._attempt_minimum_angle
            if start_frame is not None and minimum_angle is not None:
                self.completed.append(
                    CompletedSquat(
                        index=len(self.completed) + 1,
                        start_frame=start_frame,
                        end_frame=frame_index,
                        minimum_knee_angle=minimum_angle,
                    )
                )
            self._reset_attempt()
            self.phase = SquatPhase.STANDING

    def _reset_attempt(self) -> None:
        self._attempt_start_frame = None
        self._attempt_minimum_angle = None

    def _result(
        self,
        frame_index: int,
        side: BodySide | None,
        knee_angle: float | None,
        hip_angle: float | None,
        measurement_valid: bool,
    ) -> SquatFrameResult:
        return SquatFrameResult(
            frame_index=frame_index,
            phase=self.phase,
            side=side,
            knee_angle=knee_angle,
            hip_angle=hip_angle,
            completed_reps=len(self.completed),
            incomplete_attempts=self.incomplete_attempts,
            measurement_valid=measurement_valid,
        )


class SquatVideoAnnotator:
    """Estimate pose, analyze squat movement, and annotate each frame."""

    PANEL_COLOR = (24, 31, 38)
    TEXT_COLOR = (255, 255, 255)

    def __init__(
        self,
        estimator: PoseEstimator,
        *,
        analyzer: SquatAnalyzer | None = None,
        keypoint_confidence: float = 0.5,
    ) -> None:
        self.estimator = estimator
        self.analyzer = analyzer or SquatAnalyzer(
            minimum_confidence=keypoint_confidence
        )
        self.keypoint_confidence = keypoint_confidence
        self.frames_processed = 0
        self.frames_with_measurement = 0

    def __call__(self, frame: Frame, frame_index: int) -> Frame:
        pose = self.estimator.estimate(frame)
        result = self.analyzer.update(pose, frame_index)
        self.frames_processed += 1

        if pose is not None:
            draw_pose(
                frame,
                pose,
                keypoint_confidence=self.keypoint_confidence,
            )
        if result.measurement_valid:
            self.frames_with_measurement += 1

        angle_text = (
            f"Knee {result.knee_angle:.0f} deg"
            if result.knee_angle is not None
            else "Knee unavailable"
        )
        lines = (
            f"Squat | {result.phase.value}",
            f"Reps {result.completed_reps} | Incomplete {result.incomplete_attempts}",
            angle_text,
        )
        cv2.rectangle(frame, (8, 8), (340, 92), self.PANEL_COLOR, -1)
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (16, 30 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                self.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
        return frame
