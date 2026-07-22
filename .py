#!/usr/bin/env python3
"""Generate all remote GPU architecture files."""

import os
from pathlib import Path

BASE = Path("remote_gpu")
BASE.mkdir(exist_ok=True)

def write_file(path: str, content: str):
    full = BASE / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    print(f"Wrote: {full}")

# --------------------------------------------------------------------
# 1. Colab worker main.py (full corrected version)
# --------------------------------------------------------------------
worker_main = '''"""FastAPI server for remote GPU execution with checkpointing and model caching."""

import os
import sys
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Load .env file if present
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        logging.info(f"Loaded .env from {dotenv_path}")
except ImportError:
    pass

import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str = ""
    result_timeline: Optional[str] = None
    result_timeline_words: Optional[str] = None
    error: Optional[str] = None

JOBS_DIR = (PROJECT_ROOT / "jobs").resolve()
JOBS_DIR.mkdir(exist_ok=True)

def load_job(job_id: str) -> dict:
    p = JOBS_DIR / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404, "Job not found")
    return json.loads(p.read_text())

def save_job(job_id: str, data: dict):
    p = JOBS_DIR / f"{job_id}.json"
    p.write_text(json.dumps(data, indent=2))

def update_job(job_id: str, **kwargs):
    data = load_job(job_id)
    data.update(kwargs)
    save_job(job_id, data)

class PipelineStep:
    def __init__(self, name: str, module_path: str, args_template: list, checkpoint_key: str):
        self.name = name
        self.module_path = module_path
        self.args_template = args_template
        self.checkpoint_key = checkpoint_key
        self.options = {}

    def run(self, job_dir: Path, job_id: str, checkpoint: dict, options: dict) -> dict:
        self.options = options
        update_job(job_id, progress=0, message=self.name)
        if self.checkpoint_key in checkpoint:
            return checkpoint[self.checkpoint_key]

        args = []
        for a in self.args_template:
            if isinstance(a, Path):
                args.append(str(a.resolve()))
            elif isinstance(a, str) and a.startswith("JOBS_DIR/"):
                rel = a.replace("JOBS_DIR/", "")
                args.append(str((job_dir / rel).resolve()))
            else:
                args.append(str(a))

        cmd = [sys.executable, self.module_path] + args
        logger.info(f"Running: {' '.join(cmd)}")

        env = os.environ.copy()
        if self.options.get("hf_token"):
            env["HF_TOKEN"] = self.options["hf_token"]
        if "04_conversation_layer" in self.module_path:
            if not env.get("HF_TOKEN"):
                raise RuntimeError(
                    "Missing HuggingFace token for conversation (WhisperX diarization). "
                    "Set HF_TOKEN in .env file, environment variable, or pass options={\\"hf_token\\":\\"hf_xxx\\"}."
                )

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"""Command: {' '.join(cmd)}

cwd: {PROJECT_ROOT}

Output: {result.stdout}
""")

        output_path = None
        for i, arg in enumerate(args):
            if arg == "--output" and i + 1 < len(args):
                output_path = Path(args[i + 1]).resolve()
                break

        if output_path and output_path.exists():
            with open(output_path, 'r') as f:
                data = json.load(f)
        else:
            possible = list(job_dir.glob("*.json"))
            if possible:
                with open(possible[0], 'r') as f:
                    data = json.load(f)
            else:
                data = {"status": "completed"}

        checkpoint[self.checkpoint_key] = data
        return data

PIPELINE = [
    PipelineStep("Scene Detection", "video-brain/modules/01_scene_detector/main.py",
                 ["JOBS_DIR/video.mp4", "--output", "JOBS_DIR/scenes.json"], "scene"),
    PipelineStep("Face Detection", "video-brain/modules/02_face_layer/main.py",
                 ["JOBS_DIR/video.mp4", "JOBS_DIR/scenes.json", "--output", "JOBS_DIR/faces.json"], "faces"),
    PipelineStep("Identity Tracking", "video-brain/modules/03_identity_layer/main.py",
                 ["JOBS_DIR/video.mp4", "JOBS_DIR/faces.json", "JOBS_DIR/scenes.json", "--output", "JOBS_DIR/identities.json"], "identities"),
    PipelineStep("Conversation (WhisperX)", "video-brain/modules/04_conversation_layer/main.py",
                 ["JOBS_DIR/video.mp4", "--output", "JOBS_DIR/conversation.json", "--model", "tiny", "--device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"], "conversation"),
    PipelineStep("Timeline Fusion", "video-brain/modules/05_timeline_fusion/main.py",
                 ["JOBS_DIR/video.mp4", "JOBS_DIR/scenes.json", "JOBS_DIR/faces.json", "JOBS_DIR/identities.json", "JOBS_DIR/conversation.json", "--output", "JOBS_DIR/timeline.json"], "timeline"),
]

class PipelineRunner:
    def __init__(self, job_id: str, video_bytes: bytes, options: dict):
        self.job_id = job_id
        self.job_dir = JOBS_DIR / job_id
        self.job_dir.mkdir(exist_ok=True)
        self.video_path = (self.job_dir / "video.mp4").resolve()
        self.video_path.write_bytes(video_bytes)
        self.options = options
        self.checkpoint_file = (self.job_dir / "checkpoint.json").resolve()
        self.checkpoint = {}
        if self.checkpoint_file.exists():
            self.checkpoint = json.loads(self.checkpoint_file.read_text())

    def _save_checkpoint(self):
        self.checkpoint_file.write_text(json.dumps(self.checkpoint, indent=2))

    def _get_progress(self) -> float:
        steps = ["scene", "faces", "identities", "conversation", "timeline"]
        completed = [s for s in steps if s in self.checkpoint]
        return (len(completed) / len(steps)) * 100

    def run(self):
        update_job(self.job_id, status="running", progress=0, message="Initialising")
        try:
            self._run_pipeline()
            update_job(self.job_id, status="completed", progress=100, message="Done")
        except Exception as e:
            logger.exception("Pipeline failed")
            update_job(self.job_id, status="failed", error=str(e), progress=self._get_progress())

    def _run_pipeline(self):
        for step in PIPELINE:
            step.run(self.job_dir, self.job_id, self.checkpoint, self.options)
            self._save_checkpoint()
        tl_path = (self.job_dir / "timeline.json").resolve()
        tlw_path = (self.job_dir / "timeline_with_words.json").resolve()
        if tl_path.exists():
            shutil.copy(tl_path, tlw_path)
        update_job(self.job_id, result_timeline=str(tl_path), result_timeline_words=str(tlw_path), progress=100, message="Ready")

app = FastAPI(title="Video Brain GPU Worker")

@app.post("/jobs", response_model=JobStatus)
async def create_job(video: UploadFile = File(...), options: str = "{}", background_tasks: BackgroundTasks = None):
    job_id = str(uuid.uuid4())[:8]
    data = {"job_id": job_id, "status": "pending", "progress": 0.0, "message": "Queued", "result_timeline": None, "result_timeline_words": None, "error": None}
    save_job(job_id, data)
    video_bytes = await video.read()
    runner = PipelineRunner(job_id, video_bytes, json.loads(options))
    background_tasks.add_task(runner.run)
    return JobStatus(**data)

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str): return JobStatus(**load_job(job_id))

@app.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_status(job_id: str): return await get_job(job_id)

@app.get("/jobs/{job_id}/timeline")
async def get_timeline(job_id: str):
    data = load_job(job_id)
    if not data.get("result_timeline") or not Path(data["result_timeline"]).exists():
        raise HTTPException(404, "Timeline not ready")
    return FileResponse(data["result_timeline"], filename="timeline.json")

@app.get("/jobs/{job_id}/timeline_words")
async def get_timeline_words(job_id: str):
    data = load_job(job_id)
    if not data.get("result_timeline_words") or not Path(data["result_timeline_words"]).exists():
        raise HTTPException(404, "Timeline with words not ready")
    return FileResponse(data["result_timeline_words"], filename="timeline_with_words.json")

@app.post("/shutdown")
async def shutdown(): os._exit(0)

def start_tunnel():
    try:
        proc = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        import threading
        def print_url():
            for line in iter(proc.stdout.readline, ""):
                if "trycloudflare.com" in line:
                    import re
                    m = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
                    if m: print(f"\n*** PUBLIC URL: {m.group(0)} ***\n")
                print(line, end="")
        threading.Thread(target=print_url, daemon=True).start()
        return proc
    except Exception:
        logger.warning("Cloudflared not found; tunnel disabled")
        return None

if __name__ == "__main__":
    tunnel = start_tunnel()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        if tunnel: tunnel.terminate()
'''
write_file("colab_worker/main.py", worker_main)

# --------------------------------------------------------------------
# 2. Other files (minimal but complete)
# --------------------------------------------------------------------
write_file("colab_worker/requirements.txt", """fastapi>=0.110.0
uvicorn>=0.27.0
python-multipart>=0.0.6
requests>=2.31.0
pydantic>=2.0.0
pyscenedetect>=0.6.0
opencv-python>=4.5.0
insightface>=0.7.0
onnxruntime>=1.15.0
whisperx>=3.8.0
torch>=2.0.0
torchaudio>=2.0.0
huggingface-hub>=0.20.0
scikit-learn>=1.3.0
cloudflared>=0.1.0
typer>=0.9.0
rich>=13.0.0
python-dotenv>=1.0.0
""")

write_file("local_client/client.py", '''"""Local client for Video Brain GPU worker."""

import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any

class RemoteGPUClient:
    def __init__(self, base_url: str, timeout: int = 3600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def submit_job(self, video_path: str, options: Optional[Dict] = None) -> str:
        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)
        with open(video_path, "rb") as f:
            files = {"video": f}
            data = {"options": json.dumps(options or {})}
            resp = self.session.post(f"{self.base_url}/jobs", files=files, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()["job_id"]

    def get_status(self, job_id: str) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/jobs/{job_id}/status")
        resp.raise_for_status()
        return resp.json()

    def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> Dict[str, Any]:
        while True:
            status = self.get_status(job_id)
            if status["status"] in ("completed", "failed"):
                return status
            time.sleep(poll_interval)

    def download_timeline(self, job_id: str, output_dir: str = ".") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fname, endpoint in [("timeline.json", f"/jobs/{job_id}/timeline"), ("timeline_with_words.json", f"/jobs/{job_id}/timeline_words")]:
            resp = self.session.get(f"{self.base_url}{endpoint}", stream=True)
            resp.raise_for_status()
            p = out / fname
            with open(p, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        return out

    def shutdown(self): self.session.post(f"{self.base_url}/shutdown")
''')

write_file("local_client/config.py", """from dataclasses import dataclass
import os

@dataclass
class ClientConfig:
    base_url: str = os.environ.get("REMOTE_GPU_URL", "https://xxxx.trycloudflare.com")
    timeout: int = int(os.environ.get("REMOTE_GPU_TIMEOUT", "3600"))
    poll_interval: int = int(os.environ.get("REMOTE_GPU_POLL_INTERVAL", "5"))
    output_dir: str = os.environ.get("REMOTE_GPU_OUTPUT", "./outputs")
""")

write_file("start_colab.py", '''#!/usr/bin/env python3
import subprocess
import sys
import os

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "colab_worker/requirements.txt"], check=True)
    os.chdir("colab_worker")
    subprocess.run([sys.executable, "main.py"], check=True)
''')

write_file("colab_notebook.ipynb", '''{
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Video Brain Remote GPU Worker\\n","Run all cells in order. This notebook will start a GPU worker with a public URL."]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["!pip install -q fastapi uvicorn pyngrok cloudflared requests python-multipart\\n","!pip install -q pyscenedetect opencv-python insightface onnxruntime pydantic typer rich\\n","!pip install -q whisperx torch torchaudio huggingface-hub scikit-learn\\n","!pip install -q scenedetect python-dotenv"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["import os\\n","os.environ['HF_TOKEN'] = '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'"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["!python remote_gpu/start_colab.py"]}
 ],
 "metadata":{"accelerator":"GPU","colab":{"provenance":[]},"kernelspec":{"display_name":"Python 3","name":"python3"}},
 "nbformat":4,"nbformat_minor":0
}''')

write_file("README.md", """# Remote GPU Execution for Video Brain

## Setup

1. **Colab**: Upload the `remote_gpu` folder to Colab, run `colab_notebook.ipynb`.
2. **Local**: Set `REMOTE_GPU_URL` to the public URL printed in Colab.

## Usage

```python
from remote_gpu.local_client.client import RemoteGPUClient
client = RemoteGPUClient("https://xxxx.trycloudflare.com")
job_id = client.submit_job("my_video.mp4")
client.wait_for_completion(job_id)
client.download_timeline(job_id, "./results")