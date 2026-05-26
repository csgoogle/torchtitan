## This is a RL poc on TPU.

#### Dry run with pretraining:
```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train  \
  --module=torchtitan.experiments.tpu.qwen3 \
  --config=qwen3_debugmodel 
```


#### Setup env
First install vllm with TPU, following https://github.com/google-pytorch/torchtpu-vllm#2-install-torchtpu-vllm-and-dependencies

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