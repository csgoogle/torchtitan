## TorchTPU Support for TorchTitan

This directory provides a demonstration of running TorchTitan on Google Cloud TPUs using the TorchTPU framework. This document describes how to provision Cloud TPU resources and initiate TorchTitan training jobs.

---

## Before you begin

Before you start, complete the following steps:

1. **Verify your project:** Ensure you have a Google Cloud project with billing enabled.
2. **Request TPU access:** To obtain access to Cloud TPU resources, contact your Google Cloud account team.
3. **Install the gcloud CLI:** Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).

## Single-host model training on a TPU VM

For debugging and development, you can train small models directly on TPU VMs. You can create a TPU VM and connect to it using SSH.

### Prerequisites

1. **Project setup:** [Set up a Google Cloud project for Cloud TPUs](https://www.google.com/search?q=https://cloud.google.com/tpu/docs/setup-google-cloud-project).
2. **Planning:** [Plan your Cloud TPU resources](https://www.google.com/search?q=https://cloud.google.com/tpu/docs/plan-tpu-resources).
3. **Environment variables:** Set the following variables (the following example uses a `v6e` TPU):
4. **Enable the API:** Enable the [Cloud TPU API](https://docs.cloud.google.com/tpu/docs/setup-gcp-account).

### Create a TPU VM

You can create a TPU VM using the **Create TPU** page in the [Google Cloud console](https://console.cloud.google.com/compute/tpus/add) or by using the following gcloud CLI commands.

```bash
export TPU_NAME=your-tpu-name
export PROJECT_ID=your-project
export ZONE=your-zone  # For example, us-central2-b
export ACCELERATOR_TYPE=your-accelerator # For example, v6e-4
export VERSION=matching-runtime # v2-alpha-tpuv6e for v6e

```

To create a TPU VM, run the following command:

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$VERSION

```

Consult the **Quotas** page in the Google Cloud console to identify zones where Cloud TPUs are available and where your project has sufficient quota: [https://console.cloud.google.com/apis/api/tpu.googleapis.com/quotas](https://www.google.com/search?q=https://console.cloud.google.com/apis/api/tpu.googleapis.com/quotas)

**Flags:**

* `--zone`: Specifies the zone for the Cloud TPU. For more information on availability, see [Cloud TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones) and your project's **Quotas** page.
* `--accelerator-type`: Specifies the TPU version and size (for example, `v6e-4`).
* `--version`: Specifies the Cloud TPU software version (runtime). Ensure you use the correct version for your hardware. For a complete list, see [Cloud TPU software versions](https://docs.cloud.google.com/tpu/docs/runtimes#tpu_software_versions).
* `--reserved`: Use this flag if you are creating the TPU VM from a reservation.
* `--spot`: Use this flag to create a TPU Spot VM.

After you create the TPU VM, connect to it using SSH:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE  \
    --project=$PROJECT_ID

```

### Install TorchTPU

You can either build TorchTPU from source or install a prebuilt wheel.

Build instructions are available in the [TorchTPU GitHub repository](https://github.com/google-ml-infra/torch_tpu?tab=readme-ov-file#build-from-source) (note that this is a gated repository).

First, create and activate a Python 3.12 virtual environment:

```bash
sudo apt install python3.12  # Install Python 3.12 if it is not already installed.
mkdir wheel; cd wheel
python3.12 -m venv venv; source venv/bin/activate

```

#### Install a prebuilt wheel

Install dependencies. Some dependencies have to be installed before
TorchTPU.

```bash
# Package installation sequence matters here due to: https://github.com/openxla/tokamax/issues/240
pip install "jax[tpu]==0.9.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install git+https://github.com/openxla/tokamax.git@8cba6a6a1e52e9efbb7ff8facb66f18f0bfcbe4c
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install libtpu==0.0.37 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

In order to install prebuilt TorchTPU wheel, you need to authenticate first.

```bash
pip install keyrings.google-artifactregistry-auth
gcloud auth login
gcloud auth application-default login

pip install --pre --index-url --no-deps "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" --no-deps torch_tpu
```

### Install TorchTitan and its dependencies

```bash
git clone https://github.com/google-pytorch/torchtitan   # This repository will be available at a later date.
cd torchtitan
pip install -r requirements.txt
pip install -r .ci/docker/requirements-flux.txt
pip install portpicker absl-py numpy gcsfs frozendict triton fairscale

```

### Optional: Download the Hugging Face model tokenizer

This step is optional if you are running a dummy model. Follow the instructions in the [TorchTitan README](https://github.com/pytorch/torchtitan/tree/main?tab=readme-ov-file#downloading-a-tokenizer).

```bash
# Obtain your Hugging Face token from https://huggingface.co/settings/tokens

# Download the Llama 3.2 tokenizer
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.2-1B --assets tokenizer --hf_token=...

```

### Run a TorchTitan training job

Only small models can run on a single host. The following example demonstrates Llama 3.2 1B running on a `v6e-4` VM using FSDP:

```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train  \
    --job.config_file=./torchtitan/experiments/tpu/llama3/train_configs/llama3.2_1b.toml \
    --model.name=llama3_tpu \
    --model.hf_assets_path=assets/hf/Llama-3.2-1B/ \
    --training.seq_len=128 \
    --training.steps=10 \
    --training.dtype=bfloat16 \
    --training.mixed_precision_param=bfloat16

```

**Notes:**

* Currently, only the `llama3_tpu` configuration is supported.
* You must limit `seq_len` to 128 at this time; support for the full context length is coming soon.
* Memory and Model Flops Utilization (MFU) metrics might be inaccurate in the current version.

---

## Multi-host model training on GKE

To train larger models, use a multi-host configuration on Google Kubernetes Engine (GKE).

### Provision Cloud TPU resources

To provision Cloud TPUs, use one of the following methods:

* **GKE:** Use GKE to provision and manage Cloud TPUs as a pool of accelerators for containerized machine learning workloads. Use the gcloud CLI to create GKE cluster instances manually for precise customization or to expand existing production GKE environments. For more information, see [TPUs in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus).
* **GKE and XPK:** The Accelerated Processing Kit (XPK) is a command-line tool that simplifies cluster creation and workload execution on GKE. It is designed for ML practitioners to provision Cloud TPUs and run training jobs without requiring deep Kubernetes expertise. Use XPK to quickly create GKE clusters and run workloads for proof-of-concept testing. For more information, see the [XPK GitHub repository](https://github.com/google/xpk).
* **GKE and TPU Cluster Director:** TPU Cluster Director is available through an All Capacity mode reservation, providing full access to reserved capacity without hold-backs. This mode offers full visibility into TPU hardware topology, utilization, and health status. For more information, see the [All Capacity mode overview](https://www.google.com/search?q=https://docs.cloud.google.com/tpu/docs/all-capacity-mode-overview).

In this document, you use GKE managed by XPK. XPK is a command-line tool designed to simplify provisioning, managing, and running machine learning workloads.

### Install XPK and its dependencies

1. **Install XPK:** Follow the installation instructions in the [XPK GitHub repository](https://github.com/google/xpk).

```bash
pip install xpk

```

2. **Install Docker:** Follow the official Docker installation instructions. After installation, run the following commands to configure Docker and verify the installation:

```bash
gcloud auth configure-docker
sudo usermod -aG docker $USER
# Restart the terminal and reactivate the virtual environment after running this command.
docker run hello-world # Verify the Docker installation.

```

3. **Set environment variables:**

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export ACCELERATOR_TYPE=YOUR_ACCELERATOR_TYPE
export BUCKET_NAME="gs://YOUR_BUCKET_NAME"
export REPOSITORY=YOUR_REPOSITORY

```

Replace the following placeholders:

* `YOUR_PROJECT_ID`: Your Google Cloud project ID.
* `YOUR_ZONE`: The zone where the cluster will be created. For TPU availability, see [Cloud TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones) and your project's **Quotas** page.
* `YOUR_CLUSTER_NAME`: The name of the new GKE cluster.
* `YOUR_ACCELERATOR_TYPE`: The TPU version and topology (for example, `v6e-32`).
* `YOUR_BUCKET_NAME`: The name of your Cloud Storage bucket for output.
* `YOUR_REPOSITORY`: The name of your Artifact Registry repository (for example, `test`).

4. **Optional: Create a Cloud Storage bucket:**

```bash
gcloud storage buckets create ${REPOSITORY} \
    --project=${PROJECT_ID} \
    --location=US \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access

```

### Create a single-NIC, single-slice cluster

We recommend using a custom network with an MTU of 8,896 for optimal performance.

#### Option 1: Custom network (Recommended)

1. Set the network environment variables:

```bash
export NETWORK_NAME=YOUR_NETWORK_NAME
export NETWORK_FW_NAME=YOUR_FIREWALL_NAME

```

2. Create the network and firewall rule:

```bash
gcloud compute networks create ${NETWORK_NAME} \
    --mtu=8896 \
    --project=${PROJECT_ID} \
    --subnet-mode=auto \
    --bgp-routing-mode=regional

gcloud compute firewall-rules create ${NETWORK_FW_NAME} \
    --network=${NETWORK_NAME} \
    --allow tcp,icmp,udp \
    --project=${PROJECT_ID}

```

3. Create the cluster with optimized network settings:

```bash
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"

xpk cluster create --cluster=${CLUSTER_NAME} \
    --cluster-cpu-machine-type=n1-standard-8 \
    --num-slices=1 \
    --tpu-type=${ACCELERATOR_TYPE} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --on-demand \
    --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"

```

#### Option 2: Default network

Create the cluster using default network settings:

```bash
xpk cluster create --cluster=${CLUSTER_NAME} \
    --cluster-cpu-machine-type=n1-standard-8 \
    --num-slices=1 \
    --tpu-type=${ACCELERATOR_TYPE} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --on-demand

```

### Build the TorchTitan TPU-enabled Docker image

Clone the repository from GitHub (note that this repository will be available at a later date).

```bash
git clone https://github.com/google-pytorch/torchtitan
cd torchtitan

```

#### Download the TorchTPU wheel file

Before the public release of TorchTPU, you must either copy a prebuilt wheel file or build one from source.

Build instructions are available in the [TorchTPU GitHub repository](https://github.com/google-ml-infra/torch_tpu?tab=readme-ov-file#build-from-source) (note that this is a gated repository).

To download a prebuilt wheel, run the following:

```bash
pip install keyrings.google-artifactregistry-auth
gcloud auth login
gcloud auth application-default login

pip download --pre --no-deps \
    --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
    torch_tpu

```

The downloaded file will have a name similar to `torch_tpu-0.1.0.dev2242026-cp312-cp312-manylinux_2_31_x86_64.whl`. Rename it to `torch_tpu-0.1.0.dev0-cp312-cp312-manylinux_2_31_x86_64.whl` to match the hardcoded name in the Dockerfile.

#### Optional: Download the Hugging Face model tokenizer

This step is optional if you are running a dummy model. For more information, see the [TorchTitan README](https://github.com/pytorch/torchtitan/tree/main?tab=readme-ov-file#downloading-a-tokenizer).

```bash
# Obtain your Hugging Face token from https://huggingface.co/settings/tokens

# Download the Llama 3.1 tokenizer
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...

```

#### Build the Docker image

```bash
docker build -t torchtitan-container:latest -f torchtitan/experiments/tpu/Dockerfile .

```

#### Upload the Docker image to Artifact Registry

```bash
# This example assumes you are using Artifact Registry in us-central1.
# This command only needs to be run once.
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# This command only needs to be run once.
gcloud artifacts repositories create ${REPOSITORY} \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for torchtitan training"

docker tag torchtitan-container:latest us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest
docker push us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest

```

#### Optional: Attach Cloud Storage FUSE storage

To make data in Cloud Storage accessible to TorchTitan and to export logs and profiles, you can mount a bucket using Cloud Storage FUSE. This allows the bucket to appear as a local directory. For more information, see [XPK storage usage](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/storage.md).

```bash
xpk storage attach my-gcsfuse-storage \
  --type=gcsfuse \
  --project=${PROJECT_ID} \
  --cluster=${CLUSTER_NAME} \
  --zone=${ZONE} \
  --mount-point=/data \
  --readonly=false \
  --bucket=${BUCKET_NAME} \
  --size=1 \
  --auto-mount=false

```

#### Deploy the training workload

The following sample demonstrates Llama 8B training on a `v6e-32` using FSDP:

```bash
xpk workload create \
  --workload=torchtitan-llama3-8b-xpk \
  --cluster=${CLUSTER_NAME}  \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --tpu-type=${ACCELERATOR_TYPE} \
  --num-slices=1 \
  --docker-image us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
  --storage=my-gcsfuse-storage \
  --env WORKERS_0_HOSTNAME="torchtitan-llama3-8b-xpk-slice-job-0-0.torchtitan-llama3-8b-xpk" \
  --command "torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$(WORKERS_0_HOSTNAME):29501 \
    --node_rank=\$(TPU_WORKER_ID) \
    -m \
    torchtitan.experiments.tpu.train \
    --job.config_file=./torchtitan/models/llama3/train_configs/llama3_8b.toml \
    --job.dump_folder=/data/torchtitan-llama3-8b-xpk \
    --model.name=llama3_tpu \
    --training.seq_len=512 \
    --training.dataset=c4_test \
    --training.dtype=bfloat16 \
    --training.mixed_precision_param=bfloat16 \
    --training.steps=10 \
    --model.hf_assets_path=assets/hf/Llama-3.1-8B/ \
    --training.dataset_path=tests/assets/c4_test \
    --metrics.log_freq=5 \
    --metrics.enable_tensorboard \
    --profiling.profile_freq=5 \
    --profiling.profiler_warmup=3 \
    --profiling.profiler_active=1 \
    --profiling.enable-profiling"

```

**Notes:**

* This configuration was tested on `v6e-16` and `v6e-32`.
* Currently, only the `llama3_tpu` configuration is supported.
* You must limit `seq_len` to 512 at this time; support for the full context length is coming soon.
* Memory and MFU metrics might be inaccurate in the current version.

---

### Profile workloads using TensorBoard

TorchTitan on TorchTPU integrates with a profiler to export profiling data for trained models. Use the `--profiling.enable-profiling` flag to enable profiling and the `--profiling...` flags for configuration.

Profiling data is exported to the directory specified by `--job.dump_folder` + `profile_trace` for each process. In the example above, this is mapped to an external Cloud Storage bucket via Cloud Storage FUSE and can be loaded directly into TensorBoard.

For more information, see [Profile your TPU VM workload using TensorBoard](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm).

To install TensorBoard locally, run the following commands:

```bash
pip install tensorboard_plugin_profile tensorboard
pip install xprof

```

To view the profile captured for your TorchTitan training job, run:

```bash
tensorboard --logdir=gcs://${BUCKET_NAME}/torchtitan-llama3-8b-xpk/rank_0

```

### Clean up

To avoid incurring unnecessary charges, delete your resources when you are finished:

```bash
# Delete the XPK workload.
xpk workload delete --workload torchtitan-llama3-8b-xpk --cluster ${CLUSTER_NAME}

# Delete the GKE cluster.
xpk cluster delete --cluster ${CLUSTER_NAME}

```
