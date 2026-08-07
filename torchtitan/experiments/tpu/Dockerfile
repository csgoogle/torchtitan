FROM python:3.12

ARG CACHE_BUSTER=1

# Package installation sequence matters here due to: https://github.com/openxla/tokamax/issues/240
RUN pip install "jax[tpu]==0.11.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install git+https://github.com/openxla/tokamax.git@b33bdfa64a78cc16193f3c77dd223bb040aeebf4

ARG CACHE_BUSTER
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
RUN pip install libtpu==0.0.41 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install portpicker absl-py numpy gcsfs frozendict triton fairscale

# Refer to readme for instructions on building torch_tpu wheel from source with PyTorch >=2.12.

COPY torch_tpu-*.whl ./
RUN pip install --no-deps ./torch_tpu-*.whl

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .ci/docker/requirements-flux.txt .
RUN pip install -r requirements-flux.txt

COPY torchtitan /torchtitan/torchtitan
# only need to copy assets and tests when using local tokenizers and datasets
COPY assets /torchtitan/assets
COPY tests /torchtitan/tests

WORKDIR /torchtitan