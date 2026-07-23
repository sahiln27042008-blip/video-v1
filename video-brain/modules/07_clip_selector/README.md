# Clip Selector (Module 07)

Generates candidate clips from timeline and segment metrics using deterministic heuristics.

## Usage

```bash
python main.py timeline_with_words.json segment_metrics.json --output candidate_clips.json --top-k 20

`--top-k` – max candidates to return (default 20)

`--top-k`
`--min-duration` – minimum clip duration (default 2.0s)

`--min-duration`
`--max-duration` – maximum clip duration (default 60.0s)

`--max-duration`
`--merge-gap` – max gap to merge adjacent segments (default 1.0s)

`--merge-gap`
`candidate_clips.json` – sorted list of clips with scores and reasoning.

`candidate_clips.json`
Weighted combination of engagement, keywords, speech density, vocabulary richness, duration, and readability, with penalties for fillers. Weights are configurable via `config.py`.

`config.py`
Adjacent segments from the same person/speaker with gap ≤ merge_gap are merged into one candidate.