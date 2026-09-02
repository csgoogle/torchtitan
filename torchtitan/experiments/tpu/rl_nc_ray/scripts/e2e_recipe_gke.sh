#!/bin/bash
# Submit to GKE with vLLM explicitly enabled

export PATH="/home/wenjung_google_com/uv_venv/bin:$PATH"
export RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1

MODEL_SIZE=${1:-"0.6b"}
shift 

case "${MODEL_SIZE,,}" in
  "0.6b")
    CONFIG="rl_grpo_qwen3_0_6b_varlen"
    HF_ASSETS_PATH="/data/assets/assets/hf/Qwen3-0.6B"
    GEN_TP=8
    TRAINER_TP=8
    TRAINER_DP=1
    ;;
  "1.7b")
    CONFIG="rl_grpo_qwen3_1_7b_varlen"
    HF_ASSETS_PATH="/data/assets/assets/hf/Qwen3-1.7B"
    GEN_TP=1
    TRAINER_TP=2
    TRAINER_DP=4
    ;;
  "4b")
    CONFIG="rl_grpo_qwen3_4b_varlen"
    HF_ASSETS_PATH="/data/assets/assets/hf/Qwen3-4B"
    GEN_TP=8
    TRAINER_TP=8
    TRAINER_DP=1
    ;;
  *)
    echo "Unknown model size: $MODEL_SIZE. Supported: 0.6b, 1.7b, 4b"
    exit 1
    ;;
esac

ray job submit \
  --address="http://127.0.0.1:8265" \
  --working-dir . \
  --runtime-env-json '{"env_vars": {"PYTHONPATH": "torchtitan/experiments/tpu/rl_nc_ray/dummy_modules:.", "PYTHONUNBUFFERED": "1", "TORCHDYNAMO_RECOMPILE_LIMIT": "100", "TORCH_DYNAMO_RECOMPILE_LIMIT": "100", "TORCHDYNAMO_FAIL_ON_RECOMPILE_LIMIT_HIT": "0", "TORCH_DYNAMO_FAIL_ON_RECOMPILE_LIMIT_HIT": "0", "TORCHDYNAMO_CACHE_SIZE_LIMIT": "100"}, "excludes": ["assets/**", "outputs/**", "*.log", "*.output", ".git/**"]}' \
  -- \
  python3 -m torchtitan.experiments.tpu.rl_nc_ray.grpo \
    --module alphabet_sort \
    --config "$CONFIG" \
    --generator.parallelism.tensor-parallel-degree $GEN_TP \
    --generator.parallelism.spmd_backend spmd_types \
    --trainer.parallelism.tensor-parallel-degree $TRAINER_TP \
    --trainer.parallelism.data_parallel_shard_degree $TRAINER_DP \
    --hf_assets_path "$HF_ASSETS_PATH" \
    --compile.no-enable \
    --metrics.no-enable-wandb \
    --async_loop.num_training_steps 10 \
    "$@"