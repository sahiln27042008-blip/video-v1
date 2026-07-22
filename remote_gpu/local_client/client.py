"""Local client for Video Brain GPU worker."""

import os
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

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
            resp = self.session.post(
                f"{self.base_url}/jobs",
                files=files,
                data=data,
                timeout=30
            )
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
        for fname, endpoint in [
            ("timeline.json", f"/jobs/{job_id}/timeline"),
            ("timeline_with_words.json", f"/jobs/{job_id}/timeline_words"),
        ]:
            resp = self.session.get(f"{self.base_url}{endpoint}", stream=True)
            resp.raise_for_status()
            p = out / fname
            with open(p, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        return out

    def shutdown(self):
        self.session.post(f"{self.base_url}/shutdown")