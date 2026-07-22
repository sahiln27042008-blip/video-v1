# Face Layer Module

Detects faces in each scene of a video using InsightFace.

## Usage

```bash
python main.py detect <video.mp4> <scenes.json> [--output faces.json] [--threshold 0.5] [--sample-interval 0.5]

JSON with per-scene face detections (bbox, confidence).