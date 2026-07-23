# Renderer (Module 10)

FFmpeg-based video renderer that executes the AI-generated edit plan.

## Usage

```bash
python main.py video.mp4 editor_plan.json candidate_clips.json timeline_with_words.json --output ./output --music music.mp3

`--output` – output directory

`--output`
`--music` – optional background music file

`--music`
`--temp` – temporary directory for intermediate files

`--temp`
`--verbose` – show logs

`--verbose`
Clip cutting (trim)

Concatenation

Burned subtitles (ASS)

Zoom (punch-in/out)

Fade in/out

Background music (with ducking)

Audio normalization

`final_video.mp4`

`final_video.mp4`
`generated_subtitles.ass` (if created)

`generated_subtitles.ass`
`render_log.json` (if needed)

`render_log.json`
Motion graphics

Emoji

B-roll

Color grading

OpenTimelineIO

Premiere XML