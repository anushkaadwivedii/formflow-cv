"""Command line entry point for the current FormFlow video pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from formflow.curl import CurlVideoAnnotator
from formflow.pose_estimation import (
    PoseAnnotator,
    PoseEstimationError,
    YoloPoseEstimator,
)
from formflow.squat import SquatVideoAnnotator
from formflow.summary import (
    RepetitionSummary,
    SessionSummary,
    measurement_coverage,
)
from formflow.video import VideoProcessingError, process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a video through the FormFlow pipeline."
    )
    parser.add_argument("input", type=Path, help="Path to the input video")
    parser.add_argument("output", type=Path, help="Path for the processed video")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for a quick sample run",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Pose model file or name (default: yolo11n-pose.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum person confidence (default: 0.35)",
    )
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for drawing a keypoint (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device such as cpu or mps",
    )
    parser.add_argument(
        "--exercise",
        choices=("pose", "squat", "curl"),
        default="squat",
        help="Analysis mode (default: squat)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="JSON summary path (default: output video path with .json extension)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        estimator = YoloPoseEstimator(
            args.model,
            confidence=args.confidence,
            device=args.device,
        )
        if args.exercise == "squat":
            annotator = SquatVideoAnnotator(
                estimator,
                keypoint_confidence=args.keypoint_confidence,
            )
        elif args.exercise == "curl":
            annotator = CurlVideoAnnotator(
                estimator,
                keypoint_confidence=args.keypoint_confidence,
            )
        else:
            annotator = PoseAnnotator(
                estimator,
                keypoint_confidence=args.keypoint_confidence,
            )
        stats = process_video(
            args.input,
            args.output,
            transform=annotator,
            max_frames=args.max_frames,
        )
    except (ValueError, PoseEstimationError, VideoProcessingError) as error:
        raise SystemExit(f"FormFlow error: {error}") from error

    print(
        f"Processed {stats.frames_processed} frames in "
        f"{stats.elapsed_seconds:.2f}s "
        f"({stats.processing_fps:.1f} processing FPS)."
    )
    if isinstance(annotator, SquatVideoAnnotator):
        print(f"Completed squats: {len(annotator.analyzer.completed)}")
        print(f"Incomplete attempts: {annotator.analyzer.incomplete_attempts}")
        repetitions = tuple(
            RepetitionSummary(
                index=rep.index,
                start_frame=rep.start_frame,
                end_frame=rep.end_frame,
                duration_frames=rep.duration_frames,
                duration_seconds=rep.duration_frames / stats.source_fps,
                minimum_joint_angle=rep.minimum_knee_angle,
            )
            for rep in annotator.analyzer.completed
        )
        summary = SessionSummary(
            exercise="squat",
            frames_processed=annotator.frames_processed,
            frames_with_measurement=annotator.frames_with_measurement,
            measurement_coverage=measurement_coverage(
                annotator.frames_with_measurement,
                annotator.frames_processed,
            ),
            processing_seconds=stats.elapsed_seconds,
            processing_fps=stats.processing_fps,
            completed_reps=len(repetitions),
            incomplete_attempts=annotator.analyzer.incomplete_attempts,
            repetitions=repetitions,
        )
    elif isinstance(annotator, CurlVideoAnnotator):
        print(f"Completed curls: {len(annotator.analyzer.completed)}")
        print(f"Incomplete attempts: {annotator.analyzer.incomplete_attempts}")
        repetitions = tuple(
            RepetitionSummary(
                index=rep.index,
                start_frame=rep.start_frame,
                end_frame=rep.end_frame,
                duration_frames=rep.duration_frames,
                duration_seconds=rep.duration_frames / stats.source_fps,
                minimum_joint_angle=rep.minimum_elbow_angle,
            )
            for rep in annotator.analyzer.completed
        )
        summary = SessionSummary(
            exercise="curl",
            frames_processed=annotator.frames_processed,
            frames_with_measurement=annotator.frames_with_measurement,
            measurement_coverage=measurement_coverage(
                annotator.frames_with_measurement,
                annotator.frames_processed,
            ),
            processing_seconds=stats.elapsed_seconds,
            processing_fps=stats.processing_fps,
            completed_reps=len(repetitions),
            incomplete_attempts=annotator.analyzer.incomplete_attempts,
            repetitions=repetitions,
        )
    else:
        print(
            f"Pose detected in {annotator.frames_with_pose}/"
            f"{annotator.frames_processed} frames."
        )
        summary = SessionSummary(
            exercise="pose",
            frames_processed=annotator.frames_processed,
            frames_with_measurement=annotator.frames_with_pose,
            measurement_coverage=measurement_coverage(
                annotator.frames_with_pose,
                annotator.frames_processed,
            ),
            processing_seconds=stats.elapsed_seconds,
            processing_fps=stats.processing_fps,
            completed_reps=0,
            incomplete_attempts=0,
            repetitions=(),
        )
    summary_path = args.summary or args.output.with_suffix(".json")
    summary.write_json(summary_path)
    print(f"Summary: {summary_path}")
    print(f"Output: {stats.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
