#!/bin/bash
# This script builds TorchTitan Docker image for GPU using the base image
# built from .ci/docker/Dockerfile and exports TorchTitan OSS code from your
# Fig workspace. This is expected to be run from G3/Cloudtop.
#
# Usage:
# bash /path/redacted <YOUR_HF_TOKEN>

set -exu

HF_TOKEN="${1:-${HF_TOKEN:-}}"

if [ -z "$HF_TOKEN" ]; then
    echo "Error: Need to provide an HF_TOKEN as an argument or environment variable to download Llama 3 assets"
    echo "Usage: $0 <HF_TOKEN>"
    exit 1
fi

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${DIR}/../.."

CI_IMAGE_NAME="torchtitan-ubuntu-20.04-clang12"
FINAL_IMAGE_TAG="us-west1-docker.pkg.dev/tpu-pytorch/torchtitan-images/torchtitan-gpu:latest"

echo "=========================================================="
echo "Step 1: Building base CI image ${CI_IMAGE_NAME}"
echo "=========================================================="
pushd "${WORKSPACE_DIR}/.ci/docker"
bash build.sh "${CI_IMAGE_NAME}" -t "${CI_IMAGE_NAME}:latest"
popd

echo "=========================================================="
echo "Step 2: Exporting Google3 code to OSS format using Copybara"
echo "=========================================================="
# Remove any previous exports
rm -rf /tmp/torchtitan_oss
# Use copybara to apply OSS transformations then we build docker image
/google/bin/releases/copybara/public/copybara/copybara "${WORKSPACE_DIR}/copy.bara.sky" to_folder third_party/py/torchtitan --folder-dir /tmp/torchtitan_oss --ignore-noop

echo "=========================================================="
echo "Step 3: Building final image with OSS torchtitan source"
echo "=========================================================="
docker build \
    --build-arg BASE_IMAGE="${CI_IMAGE_NAME}:latest" \
    --build-arg HF_TOKEN="${HF_TOKEN}" \
    -t "${FINAL_IMAGE_TAG}" \
    -f "${DIR}/Dockerfile" \
    /tmp/torchtitan_oss

echo "=========================================================="
echo "Step 4: Pushing final image"
echo "=========================================================="
docker push "${FINAL_IMAGE_TAG}"

echo "=========================================================="
echo "Step 5: Cleanup"
echo "=========================================================="
rm -rf /tmp/torchtitan_oss

echo "Done! The fully baked image is available at: ${FINAL_IMAGE_TAG}"
