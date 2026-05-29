FROM python:3.12

# Package installation sequence matters here due to: https://github.com/openxla/tokamax/issues/240
RUN pip install "jax[tpu]==0.9.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install git+https://github.com/openxla/tokamax.git@8cba6a6a1e52e9efbb7ff8facb66f18f0bfcbe4c
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install libtpu==0.0.41 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install portpicker absl-py numpy gcsfs frozendict triton fairscale

COPY torch_tpu-0.1.0.dev0-cp312-cp312-manylinux_2_31_x86_64.whl ./torch_tpu-0.1.0.dev0-cp312-cp312-manylinux_2_31_x86_64.whl
RUN pip install --no-deps ./torch_tpu-0.1.0.dev0-cp312-cp312-manylinux_2_31_x86_64.whl

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .ci/docker/requirements-flux.txt .
RUN pip install -r requirements-flux.txt

# Restore typeguard to the version tokamax pins (setup.py says ==2.13.3).
# requirements.txt's tyro silently upgrades typeguard to 4.x at install time,
# which breaks tokamax's jaxtyping-based shape annotations at import time.
# See https://github.com/openxla/tokamax/issues/240
#
# The runtime ``typeguard.TypeCheckError = TypeError`` shim that tyro 1.x's
# import path needs lives in torchtitan/experiments/tpu/__init__.py — applies
# in any environment (docker, local conda, fresh pip), not just here.
RUN pip install 'typeguard==2.13.3'

COPY torchtitan /torchtitan
# only need to copy assets and tests when using local tokenizers and datasets
COPY assets /torchtitan/assets
COPY tests /tests

WORKDIR /torchtitan
ENV PYTHONPATH=/