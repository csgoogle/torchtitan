# TPU Non-Colocated RL with Ray

This directory contains the Ray-based orchestrator and TPU-specific components to run torchtitan Reinforcement Learning (GRPO) in a non-colocated (decoupled) architecture.

In a non-colocated setup, the Policy Trainer (PyTorch FSDP) and the Rollout Generator (vLLM) run on entirely separate slices of TPU hardware. This avoids out-of-memory errors by ensuring the training weights/gradients and the KV cache do not compete for the same HBM.

The orchestrator (`grpo.py`) runs on the Ray head node, launching remote `RayTPUPolicyTrainer` and `RayVLLMGenerator` actors on their respective TPU slices, and coordinating the state dictionary transfer via Ray's object store.

## Scripts

### 1. `smoke_test_vm_tp1.sh`
- Runs a local, single-slice smoke test utilizing CPU for the Trainer and TPU for the Generator.
- Useful for fast local debugging of the Generator/vLLM loop without needing a full multi-slice Ray cluster.

### 2. `smoke_test_vm_tp4.sh`
- Similar to `tp1`, but configures the Generator to use Tensor Parallelism degree 4.
- Validates the `torchtpu-vllm` multi-process worker initialization and TP overlapping configurations locally.

### 3. `e2e_recipe_gke.sh`
- The main production script for running an end-to-end training job across a full multi-host GKE Ray cluster.
- Submits the job to the Ray cluster head node (via `--address="http://127.0.0.1:8266"`).
- Runs 8-way FSDP for the Trainer and 8-way Tensor Parallelism for the vLLM Generator.

**Prerequisites:** Deploy the Ray cluster and port-forward to the head node:
```bash
export CLUSTER_NAME="my-gke-cluster"
export REGION="my-cluster-region"
export PROJECT="my-project"

gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT --dns-endpoint

gcloud container clusters update $CLUSTER_NAME \
    --location=$REGION \
    --project=$PROJECT \
    --update-addons=RayOperator=ENABLED 

kubectl apply -f torchtitan/experiments/tpu/rl_nc_ray/scripts/ray-v6e8-3.yaml

# Wait for the pods to be running
kubectl get pods

# Port forward the Ray head service (run this in a separate terminal)
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 8266:8265 10001:10001 
```

**Usage:**
```bash
bash torchtitan/experiments/tpu/rl_nc_ray/scripts/e2e_recipe_gke.sh
```

### 4. `plot_metrics_from_log.py`
- A utility script to visualize RL training metrics (Reward, Loss, Gen Time, Train Time) from the Ray job logs.
- **Usage:**
  Save the logs from a completed or running Ray job:
  ```bash
  ray job logs <raysubmit_id> > training.log
  python3 torchtitan/experiments/tpu/rl_nc_ray/scripts/plot_metrics_from_log.py --log_file training.log --output rl_metrics.png
  ```
