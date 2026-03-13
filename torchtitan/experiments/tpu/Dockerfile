FROM python:3.12
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install libtpu==0.0.37 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install portpicker absl-py numpy gcsfs frozendict triton fairscale

COPY torch_tpu-0.1.0.dev0-cp312-cp312-manylinux_2_31_x86_64.whl ./torch_tpu-0.1.0.dev0-cp312-cp312-manylinux_2_31_x86_64.whl
RUN pip install --no-deps ./torch_tpu-0.1.0.dev0-cp312-cp312-manylinux_2_31_x86_64.whl

COPY torchtitan/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -r ./.ci/docker/requirements-flux.txt

COPY torchtitan ./torchtitan
WORKDIR ./torchtitan