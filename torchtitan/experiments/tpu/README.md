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

### Build TorchTPU wheel from source

TorchTPU offers a pre-built wheel via

```
pip install --pre --no-deps --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" torch_tpu
```

**Important Note:** However, it currently only supports torch `2.11` and is not forward compatible with later torch versions. TorchTitan requires at least torch `>=2.12`, so we must build from torch_tpu from source.

1. First, create and activate a Python 3.12 virtual environment:

```bash
sudo apt install python3.12  # Install Python 3.12 if it is not already installed.
mkdir wheel; cd wheel
python3.12 -m venv venv; source venv/bin/activate

```

2. Clone [torch_tpu](https://github.com/google-pytorch/torch_tpu)

    ```
    git clone git@github.com:google-pytorch/torch_tpu.git
    ```

3. Change the pin in `torch_tpu/pyproject.toml` to torch `2.12`

    ```
    dependencies = [
        "torch>=2.12.0", # Uprev from 2.11.0
        ...
    ]
    ```

4. Regenerate requirements lockfiles. From the `torch_tpu` root run:

    ```
    bash requirements/lock_environments.sh
    ```

5. Follow build instructions in the [TorchTPU GitHub repository](https://github.com/google-pytorch/torch_tpu?tab=readme-ov-file#build-from-source). Instructions are repeated here:

    Wheels can be built via:

    ```bash
    bazel build -c opt //ci/wheel:torch_tpu_wheel
    ```

    (Alternative) For building without Remote Execution (RBE) caching, use bazel
    command below:

    ```bash
    bazel build -c opt //ci/wheel:torch_tpu_wheel --config=no_rbe
    ```

    Note that this command will install the CUDA version of `torch`; to get the
    CPU-only build, use the `--index-url` parameter from the section on installing
    nightlies, or install the CPU build separately.

### Install TorchTitan + TorchTPU and its dependencies

Cone the torchtitan repository:

```bash
git clone https://github.com/google-pytorch/torchtitan
cd torchtitan
```

Some dependencies have to be installed before TorchTPU.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r torchtitan/experiments/tpu/requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
pip install -r .ci/docker/requirements-flux.txt
```


Install TorchTPU wheel via:

```
pip install {PATH_TO_TORCH_TPU_REPO}/bazel-bin/ci/wheel/*.whl --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
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
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.llama3 \
    --config=llama3_1b \
    --hf_assets_path=assets/hf/Llama-3.2-1B/ \
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
* **GKE and Cluster Toolkit (`gcluster`):** Cluster Toolkit (`gcluster`) is Google Cloud's modular, production-grade framework to deploy and manage AI infrastructure on GKE and submit batch workloads. It simplifies provisioning Cloud TPUs and running distributed training jobs using Kubernetes JobSet and Kueue without requiring manual YAML authoring. For more information, see the [Cluster Toolkit documentation](https://cloud.google.com/cluster-toolkit/docs/overview) and [GitHub repository](https://github.com/GoogleCloudPlatform/cluster-toolkit).
* **GKE and TPU Cluster Director:** TPU Cluster Director is available through an All Capacity mode reservation, providing full access to reserved capacity without hold-backs. This mode offers full visibility into TPU hardware topology, utilization, and health status. For more information, see the [All Capacity mode overview](https://www.google.com/search?q=https://docs.cloud.google.com/tpu/docs/all-capacity-mode-overview).

In this document, you use GKE managed by Cluster Toolkit (`gcluster`). Cluster Toolkit is Google Cloud's toolchain designed to simplify provisioning, managing, and running machine learning workloads.

### Install Cluster Toolkit (`gcluster`) and its dependencies

1. **Install Cluster Toolkit (`gcluster`):** Follow the installation instructions in the [Cluster Toolkit GitHub repository](https://github.com/GoogleCloudPlatform/cluster-toolkit):

```bash
# Detect OS and Architecture
OS_NAME=$(uname -s | tr '[:upper:]' '[:lower:]')
CT_OS=$([ "$OS_NAME" = "darwin" ] && echo "mac" || echo "$OS_NAME")
CT_ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')

# Fetch the latest release tag
CT_VERSION=$(curl -s https://api.github.com/repos/GoogleCloudPlatform/cluster-toolkit/releases/latest | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p')
if [ -z "$CT_VERSION" ]; then
  CT_VERSION="v1.103.0"
fi

mkdir -p ~/cluster-toolkit && cd ~/cluster-toolkit
curl -fSL "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/${CT_VERSION}/gcluster_bundle_${CT_OS}_${CT_ARCH}.tgz" | tar -xz

# Add gcluster to your PATH
export PATH="$HOME/cluster-toolkit:$PATH"
gcluster --version

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
export BUCKET_NAME="YOUR_BUCKET_NAME"
export REPOSITORY=YOUR_REPOSITORY

```

Replace the following placeholders:

* `YOUR_PROJECT_ID`: Your Google Cloud project ID.
* `YOUR_ZONE`: The zone where the cluster will be created. For TPU availability, see [Cloud TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones) and your project's **Quotas** page.
* `YOUR_CLUSTER_NAME`: The name of the new GKE cluster.
* `YOUR_ACCELERATOR_TYPE`: The TPU version and topology (for example, `v6e-32`).
* `YOUR_BUCKET_NAME`: The name of your Cloud Storage bucket for output (without `gs://` prefix).
* `YOUR_REPOSITORY`: The name of your Artifact Registry repository (for example, `test`).

4. **Optional: Create a Cloud Storage bucket:**

```bash
gcloud storage buckets create gs://${BUCKET_NAME} \
    --project=${PROJECT_ID} \
    --location=US \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access

```

### Create a single-NIC, single-slice cluster

We recommend using Cluster Toolkit blueprints to provision the GKE cluster with Kueue and JobSet enabled.

1. Clone the Cluster Toolkit reference repository:

```bash
git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
cd cluster-toolkit
```

2. Deploy the TPU cluster using the reference blueprint:

```bash
gcluster deploy examples/gke-tpu-v6e/gke-tpu-v6e.yaml \
    --vars project_id=${PROJECT_ID},region=${ZONE%-*},zone=${ZONE},deployment_name=${CLUSTER_NAME},machine_type=ct6e-standard-4t,tpu_topology=4x4,num_slices=1,authorized_cidr=0.0.0.0/0
```

### Build the TorchTitan TPU-enabled Docker image

Clone the repository from GitHub

```bash
git clone https://github.com/google-pytorch/torchtitan
cd torchtitan

```

#### Optional: Download the Hugging Face model tokenizer

This step is optional if you are running a dummy model. For more information, see the [TorchTitan README](https://github.com/pytorch/torchtitan/tree/main?tab=readme-ov-file#downloading-a-tokenizer).

```bash
# Obtain your Hugging Face token from https://huggingface.co/settings/tokens

# Download the Llama 3.1 tokenizer
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...

```

Upload the tokenizer to the GCS bucket

```bash
gcloud storage cp --recursive ./assets/hf gs://${BUCKET_NAME}/assets/hf
gcloud storage cp --recursive ./tests gs://${BUCKET_NAME}/tests
```

#### Build the Docker image

First, copy the `torch_tpu` wheel built in [Build and install TorchTPU wheel from source](#build-and-install-torchtpu-wheel-from-source) to the root TorchTitan directory so it is accessible from docker build context:
```bash
cp ../torch_tpu/bazel-bin/ci/wheel/torch_tpu-*.whl ./
```

Run from the root of the torchtitan repository
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

#### Storage Mounting with Cloud Storage FUSE

In Cluster Toolkit (`gcluster`), storage mounting is configured inline during workload submission via the `--mount` flag, eliminating the need for separate attachment steps.

```bash
--mount "gs://${BUCKET_NAME};/data;rw;options=implicit-dirs"
```

#### Deploy the training workload

The following sample demonstrates Llama 8B training on a `v6e-32` using FSDP:

```bash
export WORKLOAD_NAME=torchtitan-llama3-8b

# Note: Workload names must not exceed 28 characters due to Kubernetes DNS-1123 limits.
# If using GCS FUSE, configure the dataset and assets paths within the mounted folder:
# --dump_folder="/data/assets/hf/Llama-3.1-8B"
# --dataloader.dataset_path="/data/assets/hf/Llama-3.1-8B"
# --hf_assets_path="/data/assets/hf/Llama-3.1-8B"
gcluster job submit \
    --name=$WORKLOAD_NAME \
    --cluster=$CLUSTER_NAME \
    --location=$ZONE \
    --project=$PROJECT_ID \
    --compute-type=$ACCELERATOR_TYPE \
    --num-slices=1 \
    --image="us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/torchtitan-container:latest" \
    --mount="gs://${BUCKET_NAME};/data;rw;options=implicit-dirs" \
    --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
    --env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536" \
    --command 'torchrun \
        --nnodes=8 \
        --nproc_per_node=4 \
        --rdzv_backend=static \
        --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 \
        --node_rank=$TPU_WORKER_ID \
        -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.llama3 \
        --config=llama3_8b \
        --training.seq_len=512 \
        --dataloader.dataset=c4_test \
        --training.dtype=bfloat16 \
        --training.mixed_precision_param=bfloat16 \
        --training.steps=10 \
        --hf_assets_path=assets/hf/Llama-3.1-8B/ \
        --dataloader.dataset_path=tests/assets/c4_test \
        --metrics.log_freq=5 \
        --metrics.enable_tensorboard \
        --profiler.profile_freq=5 \
        --profiler.profiler_warmup=3 \
        --profiler.profiler_active=1 \
        --profiler.enable_profiling'
```

**Notes:**

* This configuration was tested on `v6e-16` and `v6e-32`.
* Currently, only the `llama3_tpu` configuration is supported.
* You must limit `seq_len` to 512 at this time; support for the full context length is coming soon.
* Memory and MFU metrics might be inaccurate in the current version.

---

### Inspect and monitor workloads

Cluster Toolkit provides subcommands to monitor, inspect, and stream logs from running workloads:

```bash
# List submitted workloads and statuses
gcluster job list --cluster=${CLUSTER_NAME} --location=${ZONE} --project=${PROJECT_ID}

# Stream real-time container logs (--main-only filters sidecar logs)
gcluster job logs $WORKLOAD_NAME --cluster=${CLUSTER_NAME} --location=${ZONE} --project=${PROJECT_ID} --follow --main-only

# Inspect job health and Kueue scheduling conditions
gcluster job inspect --name $WORKLOAD_NAME --cluster=${CLUSTER_NAME} --location=${ZONE} --project=${PROJECT_ID} --show
```

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
tensorboard --logdir=gcs://${BUCKET_NAME}/torchtitan-llama3-8b/rank_0

```

### Clean up

To avoid incurring unnecessary charges, delete your resources when you are finished:

```bash
# Cancel and delete the workload.
gcluster job cancel $WORKLOAD_NAME --cluster=${CLUSTER_NAME} --location=${ZONE} --project=${PROJECT_ID}

# Destroy the GKE cluster deployment.
gcluster destroy ${CLUSTER_NAME}

```
