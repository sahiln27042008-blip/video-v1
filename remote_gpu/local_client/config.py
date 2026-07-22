"""Client configuration."""

from dataclasses import dataclass
import os

@dataclass
class ClientConfig:
    base_url: str = os.environ.get("REMOTE_GPU_URL", "https://xxxx.trycloudflare.com")
    timeout: int = int(os.environ.get("REMOTE_GPU_TIMEOUT", "3600"))
    poll_interval: int = int(os.environ.get("REMOTE_GPU_POLL_INTERVAL", "5"))
    output_dir: str = os.environ.get("REMOTE_GPU_OUTPUT", "./outputs")