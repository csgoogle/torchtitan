## This is a RL poc on TPU.

#### Dry run with pretraining:
```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train  \
  --module=torchtitan.experiments.tpu.qwen3 \
  --config=qwen3_debugmodel 
```

#### Collocated GRPO training with vLLM:

First install vllm with TPU, following https://github.com/google-pytorch/torchtpu-vllm#2-install-torchtpu-vllm-and-dependencies

Then:
```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.rl.train_grpo \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b_glp \
    --training.steps=8 \
    --sampler.use_vllm \
    --training.local_batch_size=2
```