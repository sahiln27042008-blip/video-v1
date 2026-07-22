# Remote GPU Execution for Video Brain

## Overview

- **Worker** runs on Google Colab (GPU) and exposes a FastAPI server.
- **Client** runs locally, uploads video, fetches timeline.

## Setup

1. **Colab**: Open `colab_notebook.ipynb` and run all cells. After accepting the Hugging Face model terms, it will start the worker and print a public URL.

2. **Local**: Set `REMOTE_GPU_URL` environment variable to that URL.

3. **Usage**:
```python
from remote_gpu.local_client.client import RemoteGPUClient
client = RemoteGPUClient("https://xxxx.trycloudflare.com")
job_id = client.submit_job("my_video.mp4")
client.wait_for_completion(job_id)
client.download_timeline(job_id, output_dir="./results")

`POST /jobs` – upload video, get job_id

`POST /jobs`
`GET /jobs/{id}/status` – check progress

`GET /jobs/{id}/status`
`GET /jobs/{id}/timeline` – download timeline.json

`GET /jobs/{id}/timeline`
`GET /jobs/{id}/timeline_words` – download timeline_with_words.json

`GET /jobs/{id}/timeline_words`
`POST /shutdown` – stop the worker

`POST /shutdown`
Jobs are checkpointed after each module. If the worker crashes, rerunning the job resumes from the last successful step.

Models are cached in memory and reused across jobs.

GPU acceleration for WhisperX, InsightFace, and PyTorch.