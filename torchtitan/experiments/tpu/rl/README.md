## TorchTitan TPU GRPO Orchestrator (Ray)

This directory contains a modular, collocated, and synchroized implementation of Group Relative Policy Optimization (GRPO) built on Ray for TPU clusters. 

This implementation is highly inspired by `verl` but is tailored specifically for the constraints and extreme interconnect speed (ICI) of TPU v6e environments. It runs the **Policy Model (PyTorch FSDP)** and the **Sampler (vLLM)** collocated on the exact same TPU chips, using Ray to orchestrate the step loop.

### Architecture
*   **`ray_orchestrator.py`**: The main Ray Driver running on the CPU Head Node. Drives the PPO steps and computes correct global advantages.
*   **`ray_worker.py`**: The Ray Actor running on each TPU chip. It holds the PyTorch FSDP models, VLLM engine, and executes forward/backward passes.
*   **`ray_train.py`**: The lightweight entrypoint script.

### Running GRPO Training

#### 0. Setup env

Following `torchtitan/experiments/tpu/README.md` to install torch_tpu / torchtitan dependency

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

Clone torchtpu-vllm project
```
cd ..
git clone https://github.com/google-pytorch/torchtpu-vllm.git
cd torchtpu-vllm
```

We need to hack `torchtpu-vllm/pyproject.toml` to be use the `torch-tpu`/`libtpu` version for `dcp`
```diff
@@ -25,10 +25,10 @@ dependencies = [
     "portpicker",
     "pathwaysutils",
     "vllm==0.19.0",
-    "torch-tpu==0.1.1.dev20260515095303",
+    "torch-tpu==0.1.1.dev20260527102151",
     # Keep the PJRT runtime pinned with the torch-tpu wheel. libtpu 0.0.41
     # regresses GDN compilation with scoped VMEM OOMs in CI.
-    "libtpu==0.0.40",
+    "libtpu==0.0.41",
     "ray[default]",
     "ray[data]",
 ]
```

Now install vllm (and ray)

```bash
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"

git clone --depth 1 --branch v0.19.0 https://github.com/vllm-project/vllm.git ../vllm
sed -i '/tpu-inference/d' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.19.0 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.19.0 VLLM_TARGET_DEVICE="tpu" uv pip install -e ../vllm

# Install TorchTPU-vLLM and dependencies
uv pip install --pre -e .
```

Dry run with pretraining and verify dcp works:

```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3 \
  --config=grpo_qwen3_0_6b \
  --training.steps=2 \
  --optimizer.lr=0.0 \
  --checkpoint.enable \
  --checkpoint.folder="assets/dcp/Qwen3-0.6B" \
  --checkpoint.initial_load_path="assets/hf/Qwen3-0.6B" \
  --checkpoint.initial_load_model_only \
  --checkpoint.initial_load_in_hf \
  --checkpoint.interval=2
```

#### 1. With High-Performance vLLM Sampling
```bash
sudo fuser -k /dev/vfio/* 2>/dev/null || true
export PYTHONPATH=$PYTHONPATH:.; PYTHONUNBUFFERED=1 python torchtitan/experiments/tpu/rl/ray_train.py \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b \
    --sampler.use_vllm \
    --checkpoint.enable \
    --training.steps=10 2>&1 \
    | tee ray_train_vllm.log \
    | grep -iE "\[titan\]|@@@" 
```

Check progress and metrics
```
grep -E "Step [0-9]+: Times|Training completed|step:\s+[0-9]+" ray_train_vllm.log
```


#### 2. Alternative: Natively with PyTorch FSDP 
If vLLM is not installed or you wish to sample directly with PyTorch's native FSDP layers (slower autoregressive performance, but useful for testing):
```bash
sudo fuser -k /dev/vfio/* 2>/dev/null || true
export PYTHONPATH=$PYTHONPATH:.; PYTHONUNBUFFERED=1 python torchtitan/experiments/tpu/rl/ray_train.py \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b \
    --sampler.no-use-vllm \
    --sampler.use-separate-sampler-model \
    --checkpoint.enable \
    --training.steps=10 2>&1 \
    | tee ray_train.log \
    | grep -iE "\[titan\]|@@@" 
```

*Note: For rapid testing of the orchestration loop without waiting for slow autoregressive FSDP generation, append `--sampler.use-fake-sampler` to generate dummy completions.*


### Running Unit Tests

To run the unit tests across the newly refactored Ray Actor/Driver system, you can use Python's built-in `unittest` module from the root directory of the project:

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover -s torchtitan/experiments/tpu/rl/tests -p "test_*.py"
```
