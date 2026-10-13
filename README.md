# FormFlow

FormFlow analyzes exercise videos with pose estimation, joint geometry, and movement state machines. Its first supported exercises will be squats and bicep curls.

The project provides fitness movement summaries for demonstration and educational use. It does not provide medical advice or injury assessment.

## Planned capabilities

1. Estimate body keypoints in video frames
2. Calculate confidence-aware joint angles
3. Smooth measurements across nearby frames
4. Count squat and bicep curl repetitions
5. Identify incomplete ranges of motion using documented heuristics
6. Save an annotated demonstration video and session summary
7. Evaluate repetition counts and runtime performance

## Technology

Python, OpenCV, YOLO pose estimation, NumPy, and pytest.

## Status

The video pipeline, pose estimation, squat analyzer, and bicep curl analyzer are complete. Each analyzer uses confidence-aware joint angles, smooths frame-to-frame measurements, tracks movement phases, and counts completed or incomplete attempts.

Each run also writes a JSON summary beside the output video with measurement coverage, processing speed, completed repetitions, repetition duration, and minimum joint angle.

## Run the current milestone

Create an environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Analyze squats in a video:

```bash
formflow input.mp4 outputs/squat.mp4 --exercise squat
```

Process only the first 100 frames:

```bash
formflow input.mp4 outputs/sample.mp4 --max-frames 100
```

Analyze bicep curls:

```bash
formflow input.mp4 outputs/curl.mp4 --exercise curl
```

Run the tests:

```bash
pytest
```

Evaluation on manually labeled workout clips is the next milestone.
