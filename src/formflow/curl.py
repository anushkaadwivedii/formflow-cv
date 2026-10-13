"""Bicep curl angle measurement, phase tracking, and video annotation."""

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


class CurlPhase(str, Enum):
    UNKNOWN = "unknown"
    EXTENDED = "extended"
    CURLING = "curling"
    CONTRACTED = "contracted"
    LOWERING = "lowering"


@dataclass(frozen=True)
class CompletedCurl:
    index: int
    start_frame: int
    end_frame: int
    minimum_elbow_angle: float

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class CurlFrameResult:
    frame_index: int
    phase: CurlPhase
    side: BodySide | None
    elbow_angle: float | None
    completed_reps: int
    incomplete_attempts: int
    measurement_valid: bool


class CurlAnalyzer:
    """Convert a sequence of poses into curl phases and repetition counts."""

    LEFT_PARTS = (
        BodyPart.LEFT_SHOULDER,
        BodyPart.LEFT_ELBOW,
        BodyPart.LEFT_WRIST,
    )
    RIGHT_PARTS = (
        BodyPart.RIGHT_SHOULDER,
        BodyPart.RIGHT_ELBOW,
        BodyPart.RIGHT_WRIST,
    )

    def __init__(
        self,
        *,
        body_side: BodySide | None = None,
        minimum_confidence: float = 0.5,
        smoothing: float = 0.35,
        extended_angle: float = 120.0,
        contracted_angle: float = 55.0,
        incomplete_angle: float = 90.0,
        hysteresis: float = 5.0,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be greater than 0 and at most 1.")
        if not 0.0 < contracted_angle < incomplete_angle < extended_angle < 180.0:
            raise ValueError(
                "Expected contracted_angle < incomplete_angle < extended_angle."
            )
        if hysteresis < 0:
            raise ValueError("hysteresis cannot be negative.")

        self.body_side = body_side
        self.minimum_confidence = minimum_confidence
        self.smoothing = smoothing
        self.extended_angle = extended_angle
        self.contracted_angle = contracted_angle
        self.incomplete_angle = incomplete_angle
        self.hysteresis = hysteresis

        self.phase = CurlPhase.UNKNOWN
        self.completed: list[CompletedCurl] = []
        self.incomplete_attempts = 0
        self._smoothed_elbow_angle: float | None = None
        self._attempt_start_frame: int | None = None
        self._attempt_minimum_angle: float | None = None

    def update(self, pose: Pose | None, frame_index: int) -> CurlFrameResult:
        measurement = self._measure(pose)
        if measurement is None:
            return self._result(frame_index, None, None, False)

        side, raw_elbow_angle = measurement
        elbow_angle = self._smooth(raw_elbow_angle)
        self._update_phase(elbow_angle, frame_index)
        return self._result(frame_index, side, elbow_angle, True)

    def _measure(self, pose: Pose | None) -> tuple[BodySide, float] | None:
        if pose is None:
            return None

        if self.body_side is not None:
            parts = (
                self.LEFT_PARTS
                if self.body_side is BodySide.LEFT
                else self.RIGHT_PARTS
            )
            if not pose.all_visible(
                parts,
                minimum_confidence=self.minimum_confidence,
            ):
                return None
            shoulder, elbow, wrist = (pose[part] for part in parts)
            return self.body_side, joint_angle(shoulder, elbow, wrist)

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
            shoulder, elbow, wrist = (pose[part] for part in parts)
            return side, joint_angle(shoulder, elbow, wrist)
        return None

    def _smooth(self, elbow_angle: float) -> float:
        if self._smoothed_elbow_angle is None:
            self._smoothed_elbow_angle = elbow_angle
        else:
            self._smoothed_elbow_angle = (
                self.smoothing * elbow_angle
                + (1.0 - self.smoothing) * self._smoothed_elbow_angle
            )
        return self._smoothed_elbow_angle

    def _update_phase(self, elbow_angle: float, frame_index: int) -> None:
        if self.phase is CurlPhase.UNKNOWN:
            if elbow_angle >= self.extended_angle:
                self.phase = CurlPhase.EXTENDED
            return

        if self.phase is CurlPhase.EXTENDED:
            if elbow_angle < self.extended_angle - self.hysteresis:
                self.phase = CurlPhase.CURLING
                self._attempt_start_frame = frame_index
                self._attempt_minimum_angle = elbow_angle
            return

        if self._attempt_minimum_angle is None:
            self._attempt_minimum_angle = elbow_angle
        else:
            self._attempt_minimum_angle = min(
                self._attempt_minimum_angle,
                elbow_angle,
            )

        if self.phase is CurlPhase.CURLING:
            if elbow_angle <= self.contracted_angle:
                self.phase = CurlPhase.CONTRACTED
            elif elbow_angle >= self.extended_angle:
                if self._attempt_minimum_angle <= self.incomplete_angle:
                    self.incomplete_attempts += 1
                self._reset_attempt()
                self.phase = CurlPhase.EXTENDED
            return

        if self.phase is CurlPhase.CONTRACTED:
            if elbow_angle > self.contracted_angle + self.hysteresis:
                self.phase = CurlPhase.LOWERING
            return

        if self.phase is CurlPhase.LOWERING and elbow_angle >= self.extended_angle:
            start_frame = self._attempt_start_frame
            minimum_angle = self._attempt_minimum_angle
            if start_frame is not None and minimum_angle is not None:
                self.completed.append(
                    CompletedCurl(
                        index=len(self.completed) + 1,
                        start_frame=start_frame,
                        end_frame=frame_index,
                        minimum_elbow_angle=minimum_angle,
                    )
                )
            self._reset_attempt()
            self.phase = CurlPhase.EXTENDED

    def _reset_attempt(self) -> None:
        self._attempt_start_frame = None
        self._attempt_minimum_angle = None

    def _result(
        self,
        frame_index: int,
        side: BodySide | None,
        elbow_angle: float | None,
        measurement_valid: bool,
    ) -> CurlFrameResult:
        return CurlFrameResult(
            frame_index=frame_index,
            phase=self.phase,
            side=side,
            elbow_angle=elbow_angle,
            completed_reps=len(self.completed),
            incomplete_attempts=self.incomplete_attempts,
            measurement_valid=measurement_valid,
        )


class CurlVideoAnnotator:
    """Estimate pose, analyze curls, and annotate each frame."""

    PANEL_COLOR = (24, 31, 38)
    TEXT_COLOR = (255, 255, 255)

    def __init__(
        self,
        estimator: PoseEstimator,
        *,
        analyzer: CurlAnalyzer | None = None,
        keypoint_confidence: float = 0.5,
    ) -> None:
        self.estimator = estimator
        if analyzer is None:
            self.analyzers = (
                CurlAnalyzer(
                    body_side=BodySide.LEFT,
                    minimum_confidence=keypoint_confidence,
                ),
                CurlAnalyzer(
                    body_side=BodySide.RIGHT,
                    minimum_confidence=keypoint_confidence,
                ),
            )
        else:
            self.analyzers = (analyzer,)
        self.analyzer = self.analyzers[0]
        self.keypoint_confidence = keypoint_confidence
        self.frames_processed = 0
        self.frames_with_measurement = 0

    @property
    def completed(self) -> tuple[CompletedCurl, ...]:
        completed = sorted(
            (
                repetition
                for analyzer in self.analyzers
                for repetition in analyzer.completed
            ),
            key=lambda repetition: repetition.end_frame,
        )
        return tuple(
            CompletedCurl(
                index=index,
                start_frame=repetition.start_frame,
                end_frame=repetition.end_frame,
                minimum_elbow_angle=repetition.minimum_elbow_angle,
            )
            for index, repetition in enumerate(completed, start=1)
        )

    @property
    def incomplete_attempts(self) -> int:
        return sum(analyzer.incomplete_attempts for analyzer in self.analyzers)

    def __call__(self, frame: Frame, frame_index: int) -> Frame:
        pose = self.estimator.estimate(frame)
        results = tuple(
            analyzer.update(pose, frame_index) for analyzer in self.analyzers
        )
        self.frames_processed += 1

        if pose is not None:
            draw_pose(
                frame,
                pose,
                keypoint_confidence=self.keypoint_confidence,
            )
        if any(result.measurement_valid for result in results):
            self.frames_with_measurement += 1

        if len(results) == 1:
            result = results[0]
            angle_text = (
                f"Elbow {result.elbow_angle:.0f} deg"
                if result.elbow_angle is not None
                else "Elbow unavailable"
            )
            lines = (
                f"Curl | {result.phase.value}",
                f"Reps {len(self.completed)} | Incomplete {self.incomplete_attempts}",
                angle_text,
            )
        else:
            left, right = results
            left_angle = (
                f"{left.elbow_angle:.0f} deg"
                if left.elbow_angle is not None
                else "unavailable"
            )
            right_angle = (
                f"{right.elbow_angle:.0f} deg"
                if right.elbow_angle is not None
                else "unavailable"
            )
            lines = (
                f"Curl | L {left.phase.value} | R {right.phase.value}",
                f"Reps {len(self.completed)} | Incomplete {self.incomplete_attempts}",
                f"Left elbow {left_angle}",
                f"Right elbow {right_angle}",
            )
        panel_height = 18 + len(lines) * 25
        cv2.rectangle(frame, (8, 8), (410, panel_height), self.PANEL_COLOR, -1)
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
