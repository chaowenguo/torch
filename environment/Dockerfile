FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential tmux && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir pytest

