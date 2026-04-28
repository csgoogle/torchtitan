# This docker files is used to build final TorchTitan image for GPU with using
# base CI image built from .ci/docker/Dockerfile.

ARG BASE_IMAGE="torchtitan-ubuntu-20.04-clang12:latest"
FROM ${BASE_IMAGE}

# While TT CI images use ci-user, we switch back to root user to avoid permissions
# issues when runing NCCL init scripts and getting NCCL conf in GKE
USER root

# Create the torchtitan directory
RUN mkdir -p /workspace/torchtitan

WORKDIR /workspace/torchtitan

# Install additional optional dependencies missing from the base CI image
RUN pip install jax psutil portpicker absl-py numpy gcsfs frozendict triton fairscale

# Copy the entire torchtitan source code into the container
COPY . /workspace/torchtitan/

# Download HF Assets
ARG HF_TOKEN
RUN python3 scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token="${HF_TOKEN}"
RUN python3 scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.2-1B --assets tokenizer --hf_token="${HF_TOKEN}"
RUN python3 scripts/download_hf_assets.py --repo_id LGAI-EXAONE/K-EXAONE-236B-A23B --assets tokenizer --hf_token="${HF_TOKEN}"
