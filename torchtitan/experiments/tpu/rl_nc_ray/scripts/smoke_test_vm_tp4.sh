#!/bin/bash

# Clean up any stale TPU processes and Ray clusters to prevent driver lock deadlocks
sudo fuser -k -9 /dev/vfio/* 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

export PYTHONPATH=$PYTHONPATH:torchtitan/experiments/tpu/rl_nc_ray/dummy_modules:.
export LOCAL_VM_RUN=1
unset RAY_ADDRESS

python3 torchtitan/experiments/tpu/rl_nc_ray/grpo.py \
    --module alphabet_sort \
    --config rl_grpo_qwen3_0_6b_varlen \
    --hf_assets_path assets/hf/Qwen3-0.6B \
    --trainer.training.steps 3 \
    --trainer.parallelism.tensor_parallel_degree 1 \
    --generator.parallelism.tensor_parallel_degree 4 \
    --compile.no-enable \
    --generator.cudagraph.no-enable \
    2>&1 \
    | tee test_e2e_tp4.log \
    | grep -iE "\[titan\]|@@@"