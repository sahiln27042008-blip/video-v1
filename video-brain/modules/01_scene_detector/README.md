# Scene Detector Module

Detects scenes in a video using PySceneDetect (ContentDetector).

## Usage

```bash
python main.py detect <video.mp4> [--output scenes.json] [--threshold 30.0] [--min-scene-len 15]

JSON file with scene start/end timestamps and duration.

`pytest`
`tests/`