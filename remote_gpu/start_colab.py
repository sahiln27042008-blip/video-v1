#!/usr/bin/env python3
"""Entry point for Colab GPU worker. Installs dependencies and starts the server."""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "colab_worker/requirements.txt"], check=True)
    # Start the worker
    os.chdir("colab_worker")
    subprocess.run([sys.executable, "main.py"], check=True)