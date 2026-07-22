"""FastAPI server for remote GPU execution. Designed to run inside Google Colab."""

import os
import subprocess
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthResponse(BaseModel):
    status: str
    gpu: bool
    cuda: Optional[str] = None
    available_modules: List[str] = Field(default_factory=list)

class AnalyzeRequest(BaseModel):
    module: str  # scene, faces, identity, conversation, ocr, objects, audio
    video_path: Optional[str] = None
    scenes_json: Optional[str] = None
    faces_json: Optional[str] = None
    identities_json: Optional[str] = None
    output_path: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

class AnalyzeResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: check GPU and log
    gpu_available = False
    cuda_version = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            cuda_version = torch.version.cuda
    except ImportError:
        pass
    logger.info(f"GPU available: {gpu_available}, CUDA: {cuda_version}")
    app.state.gpu = gpu_available
    app.state.cuda = cuda_version
    yield
    # Shutdown cleanup

app = FastAPI(
    title="Video Brain GPU Server",
    description="Remote execution for AI video analysis modules.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "Video Brain GPU Server",
        "status": "running",
        "gpu": app.state.gpu,
        "cuda": app.state.cuda,
    }

@app.post("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        gpu=app.state.gpu,
        cuda=app.state.cuda,
        available_modules=["scene", "faces", "identity", "conversation", "ocr", "objects", "audio"],
    )

@app.post("/scene", response_model=AnalyzeResponse)
async def scene(request: AnalyzeRequest) -> AnalyzeResponse:
    """Placeholder for scene detection."""
    logger.info(f"Received scene request: {request}")
    # Future: call actual scene detector
    return AnalyzeResponse(status="success", result={"message": "Scene detection placeholder"})

@app.post("/faces", response_model=AnalyzeResponse)
async def faces(request: AnalyzeRequest) -> AnalyzeResponse:
    """Placeholder for face detection."""
    logger.info(f"Received faces request: {request}")
    return AnalyzeResponse(status="success", result={"message": "Face detection placeholder"})

@app.post("/identity", response_model=AnalyzeResponse)
async def identity(request: AnalyzeRequest) -> AnalyzeResponse:
    """Placeholder for identity tracking."""
    logger.info(f"Received identity request: {request}")
    return AnalyzeResponse(status="success", result={"message": "Identity tracking placeholder"})

@app.post("/conversation", response_model=AnalyzeResponse)
async def conversation(request: AnalyzeRequest) -> AnalyzeResponse:
    """Placeholder for conversation analysis."""
    logger.info(f"Received conversation request: {request}")
    return AnalyzeResponse(status="success", result={"message": "Conversation analysis placeholder"})

@app.post("/ocr", response_model=AnalyzeResponse)
async def ocr(request: AnalyzeRequest) -> AnalyzeResponse:
    """Placeholder for OCR."""
    logger.info(f"Received OCR request: {request}")
    return AnalyzeResponse(status="success", result={"message": "OCR placeholder"})

@app.post("/objects", response_model=AnalyzeResponse)
async def objects(request: AnalyzeRequest) -> AnalyzeResponse:
    """Placeholder for object detection."""
    logger.info(f"Received object detection request: {request}")
    return AnalyzeResponse(status="success", result={"message": "Object detection placeholder"})

@app.post("/audio", response_model=AnalyzeResponse)
async def audio(request: AnalyzeRequest) -> AnalyzeResponse:
    """Placeholder for audio analysis."""
    logger.info(f"Received audio request: {request}")
    return AnalyzeResponse(status="success", result={"message": "Audio analysis placeholder"})

@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    logger.info("Shutting down server...")
    # Graceful shutdown: exit the process after returning response
    import asyncio
    asyncio.create_task(_shutdown_delay())
    return {"status": "shutting down"}

async def _shutdown_delay():
    await asyncio.sleep(1)
    os._exit(0)

def start_tunnel():
    """Start Cloudflare Tunnel and print the public URL."""
    try:
        # Use cloudflared tunnel
        import subprocess
        import threading
        import time

        # We'll use a simple approach: run cloudflared tunnel --url http://localhost:8000
        # and capture the output to extract the URL.
        # In Colab, we can use the cloudflared binary installed via pip.
        # Alternatively, use pyngrok (but requires token).
        # We'll assume cloudflared is installed or available.
        cmd = ["cloudflared", "tunnel", "--url", "http://localhost:8000"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def print_output():
            for line in iter(process.stdout.readline, ""):
                if "trycloudflare.com" in line:
                    import re
                    match = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
                    if match:
                        url = match.group(0)
                        print(f"\n*** Public URL: {url} ***\n")
                sys.stdout.write(line)
                sys.stdout.flush()

        threading.Thread(target=print_output, daemon=True).start()
        return process
    except Exception as e:
        logger.error(f"Failed to start tunnel: {e}")
        return None

def main():
    """Entry point for Colab notebook."""
    # Install dependencies if needed (this will be run in Colab)
    # We assume the notebook already installed requirements.
    print("Starting FastAPI server on http://0.0.0.0:8000")
    tunnel_process = start_tunnel()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel_process:
            tunnel_process.terminate()

if __name__ == "__main__":
    main()