# AI Edit Planner (Module 09)

Uses an LLM to generate detailed editing instructions for selected clips, transforming creative decisions into a professional edit plan.

## Usage

```bash
python main.py final_clip_plan.json timeline_with_words.json people.json metrics.json segment_metrics.json candidate_clips.json --output edit_plan.json

`--provider` – `deepseek` (default) or `openai`

`--provider`
`deepseek`
`openai`
`--model` – override the default model

`--model`
`--api-key` – API key (or set `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` env var)

`--api-key`
`DEEPSEEK_API_KEY`
`OPENAI_API_KEY`
`edit_plan.json` – a list of clips with detailed editing instructions, including subtitles, zooms, sound effects, B-roll, music, transitions, and CTAs.

`edit_plan.json`
DeepSeek (via API)

OpenAI (GPT-4)

The system prompt is stored in `editor.md`. Edit this file to change the AI's behaviour without modifying code.

`editor.md`
Add a new provider by subclassing `BaseLLMProvider` in `providers.py`.

`BaseLLMProvider`
`providers.py`