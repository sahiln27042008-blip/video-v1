# AI Planner (Module 08)

Uses an LLM to select the best clips from candidate_clips.json and produce a final clip plan.

## Usage

```bash
python main.py candidate_clips.json timeline_with_words.json people.json metrics.json segment_metrics.json --output final_clip_plan.json

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
`final_clip_plan.json` – a list of selected clips with reasoning and scores.

`final_clip_plan.json`
DeepSeek (via API)

OpenAI (GPT-4)

Add a new provider by subclassing `BaseLLMProvider` in `providers.py`.

`BaseLLMProvider`
`providers.py`