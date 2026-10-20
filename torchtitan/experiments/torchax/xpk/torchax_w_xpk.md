# Running Torchax Jobs on GKE using XPK

This guide outlines the steps to set up a Google Kubernetes Engine (GKE) cluster and submit Torchax training jobs using the XPK tool.

[TOC]

## 1. Prerequisites

These steps should be performed on your Cloudtop.

### 1.1. Google Cloud Authentication

First, install `gcloud` following
the [Google Cloud CLI installation guide](https://docs.cloud.google.com/sdk/docs/install-sdk).
Then, ensure you are authenticated with gcloud:

```bash
gcloud init
gcloud auth login
gcloud auth configure-docker
gcloud config set project cloud-tpu-multipod-dev
```

### 1.2. Install Dependencies

Install required tools if they are not already present:

*   **kubectl:**

    Following instruction from https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl

*   **Python & Virtualenv:**

    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    sudo apt-get install virtualenv python3-venv
    ```

*   **Docker:**
    (Needed for building the job image later)

    ```bash
    sudo glinux-add-repo docker-ce-"$(lsb_release -cs)"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker $USER
    # Log out and log back in or run 'newgrp docker' for the group change to take effect.
    ```

For more potential prerequisites, refer to the [XPK repository](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#prerequisites).

### 1.3. Set up XPK Virtual Environment

Create and activate a virtual environment for XPK:

```bash
python3 -m venv xpk-env 
source xpk-env/bin/activate
pip install xpk
```
*(To deactivate, simply run `deactivate`)*

## 2. Create XPK Cluster

### 2.1. Environment Variables

Define environment variables for your cluster configuration:

```bash
export PROJECT=cloud-tpu-multipod-dev
export REGION=us-central2
export ZONE=us-central2-b
export CLUSTER_NAME=${USER}-test-v4-8-2slice
export NETWORK_NAME=${CLUSTER_NAME}-net
export SUBNET_NAME=${NETWORK_NAME}-subnet
# Find a suitable reservation:
# gcloud beta compute reservations list --project=${PROJECT}
export RESERVATION_NAME=cloudtpu-us-central2-b-v4-pod-735972712744
```

### 2.2. Network Setup

Create a dedicated VPC network and subnet with secondary ranges for GKE pods and services. This avoids IP range conflicts.

*   **Create VPC Network:**

    ```bash
    gcloud compute networks create ${NETWORK_NAME} \
      --project=${PROJECT} \
      --subnet-mode=custom \
      --mtu=8896
    ```

*   **Create Subnet:**
    Choose IP ranges that don't overlap with existing subnets in your project.

    ```bash
    gcloud compute networks subnets create ${SUBNET_NAME} \
      --project=${PROJECT} \
      --network=${NETWORK_NAME} \
      --range=10.192.0.0/20 \
      --secondary-range=pods=10.200.0.0/14,services=10.204.0.0/19 \
      --region=${REGION}
    ```
    *   `--range`: Primary range for nodes.
    *   `--secondary-range=pods`: Range for Pod IPs.
    *   `--secondary-range=services`: Range for Service IPs.

*   **Create Firewall Rules:**
    Allow internal traffic and SSH.

    ```bash
    gcloud compute firewall-rules create ${NETWORK_NAME}-allow-internal \
      --project=${PROJECT} \
      --network=${NETWORK_NAME} \
      --allow=tcp:0-65535,udp:0-65535,icmp \
      --source-ranges=10.192.0.0/20,10.200.0.0/14,10.204.0.0/19

    gcloud compute firewall-rules create ${NETWORK_NAME}-allow-ssh \
      --project=${PROJECT} \
      --network=${NETWORK_NAME} \
      --allow=tcp:22
    ```

### 2.3. Launch XPK Cluster

Create the GKE cluster using XPK:

```bash
xpk cluster create \
  --cluster ${CLUSTER_NAME} \
  --device-type v4-8 \
  --num-slices 2 \
  --project ${PROJECT} \
  --zone ${ZONE} \
  --default-pool-cpu-machine-type n1-standard-16 \
  --custom-cluster-arguments="--network=${NETWORK_NAME} --subnetwork=${SUBNET_NAME} --enable-ip-alias --cluster-secondary-range-name=pods --services-secondary-range-name=services --region=${REGION}" \
  --reservation ${RESERVATION_NAME}
```
*   `--enable-ip-alias` is required for GKE to use secondary ranges.
*   `--cluster-secondary-range-name=pods` tells GKE to use the range named `pods`.
*   `--services-secondary-range-name=services` tells GKE to use the range named `services`.

### 2.4. Verify Cluster

Check the cluster status:

```bash
xpk cluster describe \
  --cluster ${CLUSTER_NAME} \
  --project ${PROJECT} \
  --zone ${ZONE}
```

## 3. Submit TorchAX Job

### 3.1. Get Example Code

Clone the torchax repository (will be replaced by torchtitan later):

```bash
git clone https://github.com/google/torchax.git
```

### 3.2. Build and Push Docker Image

Build the Docker image for the training job and push it to Google Container Registry (GCR):

```bash
export IMAGE_NAME=${USER}_torchax_runner
export IMAGE_TAG=latest

docker build -f torchax/examples/train_llama_torchtitan/Dockerfile -t $IMAGE_NAME:$IMAGE_TAG .
docker tag ${IMAGE_NAME}:${IMAGE_TAG} gcr.io/${PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}

# Authenticate Docker with GCR
gcloud auth login --update-adc
gcloud auth configure-docker

docker push gcr.io/${PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}
```

### 3.3. Submit Workload

Submit the training job to the GKE cluster using XPK. Currently, this example supports single-slice workload:

```bash
xpk workload create \
  --cluster ${CLUSTER_NAME} \
  --docker-image gcr.io/${PROJECT}/${IMAGE_NAME}:${IMAGE_TAG} \
  --workload ${USER}-torchax-$(date +%Y%m%d-%H%M%S) \
  --tpu-type v4-8 \
  --num-slices 1  \
  --command "python3 examples/train_llama_torchtitan/train_llama.py --model_type=8B" \
  --project $PROJECT \
  --zone $ZONE
```

XPK will output a link to monitor the workload in the Google Cloud Console, similar to this:
`[XPK] Follow your workload here: https://console.cloud.google.com/kubernetes/service/us-central2/jialei-test-v4-8-2slice/default/jialeic-torchax-20251118-000332/details?project=cloud-tpu-multipod-dev`

### 3.4. Monitor Workload

You can list and check the status of your workloads:

```bash
xpk workload list --cluster ${CLUSTER_NAME} --zone $ZONE --project $PROJECT
```
