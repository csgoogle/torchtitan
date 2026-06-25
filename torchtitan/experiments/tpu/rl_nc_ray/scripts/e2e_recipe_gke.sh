#!/bin/bash
# Submit to GKE with vLLM explicitly enabled

export RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1

ray job submit \
  --address="http://127.0.0.1:8266" \
  --working-dir . \
  --runtime-env-json '{"env_vars": {"PYTHONPATH": "torchtitan/experiments/tpu/rl_nc_ray/dummy_modules:.", "PYTHONUNBUFFERED": "1"}}' \
  -- \
  python3 -m torchtitan.experiments.tpu.rl_nc_ray.grpo \
    --module rl \
    --config rl_grpo_qwen3_0_6b \
    --generator.parallelism.tensor-parallel-degree 8 \
    --trainer.parallelism.tensor-parallel-degree 2 \
    --trainer.parallelism.data_parallel_shard_degree 4 \
    --num_prompts_per_step 4 \
    --hf_assets_path /data/jialei/assets/hf/Qwen3-0.6B \
    --num_steps 10 \
    --log_samples \
    "$@"