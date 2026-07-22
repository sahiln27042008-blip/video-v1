"""Configuration for the GPU worker."""

import os
from dataclasses import dataclass

@dataclass
class WorkerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    jobs_dir: str = "./jobs"
    use_tunnel: bool = True
    hf_token: str = os.environ.get("HF_TOKEN", "")
    max_jobs: int = 10