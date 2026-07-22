# Metrics Layer (Module 06)

Computes 20+ metrics from `timeline_with_words.json` without rerunning transcription or diarisation.

## Usage

```bash
python main.py timeline_with_words.json --output metrics.json

Video duration

Number of scenes

Average scene duration

Total words

Words per minute

Speaking time / silence time

Speech ratio

Average sentence length

Filler word count

Readability (Flesch, Flesch-Kincaid)

Top keywords (YAKE)

Vocabulary richness

Average confidence

Longest monologue

Speakers count & speaking time per speaker

Average & longest pause

Engagement score (heuristic)

`textstat` – readability scores

`textstat`
`yake` – keyword extraction

`yake`
`nltk` – sentence tokenisation

`nltk`