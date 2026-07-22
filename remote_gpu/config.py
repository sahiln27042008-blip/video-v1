"""Configuration for remote GPU client and server."""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RemoteGPUConfig:
    """Configuration for the remote GPU infrastructure."""

    # Server settings (for client)
    server_url: str = "https://xxxx.trycloudflare.com"
    timeout: int = 30
    retries: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None

    # Server settings (for the FastAPI app)
    host: str = "0.0.0.0"
    port: int = 8000
    use_tunnel: bool = True  # Whether to start Cloudflare tunnel

    @classmethod
    def from_env(cls) -> "RemoteGPUConfig":
        """Load configuration from environment variables."""
        return cls(
            server_url=os.environ.get("REMOTE_GPU_URL", "https://xxxx.trycloudflare.com"),
            timeout=int(os.environ.get("REMOTE_GPU_TIMEOUT", "30")),
            retries=int(os.environ.get("REMOTE_GPU_RETRIES", "3")),
            retry_delay=float(os.environ.get("REMOTE_GPU_RETRY_DELAY", "1.0")),
            api_key=os.environ.get("REMOTE_GPU_API_KEY"),
        )