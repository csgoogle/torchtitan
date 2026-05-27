## This is a RL poc on TPU.

This is under active development. The following instruction is verified as of 05/26/2026.


#### Setup env

Following https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/README.md to install torch_tpu/torchtitan dependency

```bash
uv venv tpu_rl_env --python 3.12
source tpu_rl_env/bin/activate

uv pip install keyrings.google-artifactregistry-auth
gcloud auth login
gcloud auth application-default login

uv pip install "jax[tpu]==0.9.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install git+https://github.com/openxla/tokamax.git@8cba6a6a1e52e9efbb7ff8facb66f18f0bfcbe4c
uv pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install libtpu==0.0.41 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

uv pip install --pre --no-deps --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" torch_tpu

uv pip install -r requirements.txt
uv pip install -r .ci/docker/requirements-flux.txt
uv pip install portpicker absl-py numpy gcsfs frozendict triton fairscale
```

Following https://github.com/google-pytorch/torchtpu-vllm#2-install-torchtpu-vllm-and-dependencies to install vllm with TPU

Note that at this moment, vllm pinged the following versions, which dose NOT support dcp.
```
libtpu==0.0.40
torch==2.10.0+cpu
torch-tpu==0.1.1.dev20260515095303
```

Note 
```bash
git clone https://github.com/google-pytorch/torchtpu-vllm.git
cd torchtpu-vllm


export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"

git clone --depth 1 --branch v0.19.0 https://github.com/vllm-project/vllm.git ../vllm
sed -i '/tpu-inference/d' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.19.0 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.19.0 VLLM_TARGET_DEVICE="tpu" uv pip install -e ../vllm

# Install TorchTPU-vLLM and dependencies
uv pip install --pre -e .
```


#### Dry run with pretraining:
```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train  \
  --module=torchtitan.experiments.tpu.qwen3 \
  --config=grpo_qwen3_0_6b \
  --training.steps=2 
```


#### Collocated GRPO training with Torchrun:
```bash
sudo fuser -k /dev/vfio/* || true  # Clean up any lingering TPU processes
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.rl.train_grpo \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b \
    --training.steps=8 \
    --sampler.use_vllm \
    --training.local_batch_size=2
```

#### Collocated GRPO training with Ray:
```bash
sudo fuser -k /dev/vfio/* 2>/dev/null || true
export PYTHONPATH=$PYTHONPATH:.; PYTHONUNBUFFERED=1 python torchtitan/experiments/tpu/rl/ray_train.py \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b \
    --sampler.use_vllm \
    --training.steps=100 2>&1 \
    | tee ray_train.log \
    | grep -iE "\[titan\]|error|traceback|exception|fail|critical" 
```

Check progress and metrics
```
grep -E "Step [0-9]+: Avg Reward|Training completed|step:\s+[0-9]+" ray_train.log
```