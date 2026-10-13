# FormFlow CV Project

## Goal

Build a compact, credible computer vision portfolio project for exercise pose analysis.

The application should process a prerecorded exercise video, estimate body keypoints, calculate joint angles, count repetitions, annotate the output video, and report measured runtime performance.

## Scope

Keep the first version deliberately small and complete.

Required:

1. Python video processing pipeline
2. Pretrained pose estimator
3. Confidence-aware joint geometry
4. Squat and bicep curl analyzers
5. Annotated output video and session summary
6. Basic evaluation and performance report
7. Automated tests for project-owned logic
8. Clear README and reproducible commands

Do not add medical claims, injury assessment, custom model training, automatic exercise classification, or a large frontend unless the core version is finished and the user explicitly approves the expansion.

## Working style

Explain computer vision concepts in plain language as they are introduced. Keep claims honest and record measured results before drafting resume bullets.

Work in small Git commits. Each commit should represent a working checkpoint that can be pushed to GitHub. Do not commit downloaded videos, model weights, generated output videos, virtual environments, or secrets.

## Intended stack

Python, OpenCV, Ultralytics YOLO pose, NumPy, pytest, and GitHub Actions.
