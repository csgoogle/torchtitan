# Must match PYTHON_VERSION in .github/workflows/nightly-torchtitan-image.yml,
# which is what the torch_tpu wheel is built against; the workflow passes it as
# a build arg. The default only serves manual `docker build` invocations.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}

ARG CACHE_BUSTER=1

# Install PyTorch CPU nightly first so downstream packages do not pull CUDA wheels
ARG CACHE_BUSTER
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu

# Install TPU requirements
COPY torchtitan/experiments/tpu/requirements.txt ./requirements-tpu.txt
RUN pip install -r ./requirements-tpu.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

COPY torch_tpu-*.whl ./
RUN pip install ./torch_tpu-*.whl

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .ci/docker/requirements-flux.txt .
RUN pip install -r requirements-flux.txt

# Install test utilities (CI runs with this image)
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY torchtitan /torchtitan/torchtitan
COPY assets /torchtitan/assets
COPY tests /torchtitan/tests

WORKDIR /torchtitan
