#!/usr/bin/env bash
# Copies what a rebuild of the TPU image needs from an existing image into the
# current directory (the docker build context), so no bazel build or HF token
# is needed:
#   - the torch_tpu wheel
#   - the tokenizers the nightly downloads into assets/hf
# and emits the torch nightly the wheel was built against (torch_version) and
# the image's python_version, for the Dockerfile's build args.
#
# Usage: extract_base.sh <image ref>
# Outputs go to $GITHUB_OUTPUT (stdout when run locally).
set -euo pipefail

base_ref="$1"
out="${GITHUB_OUTPUT:-/dev/stdout}"

docker pull -q "${base_ref}"
cid=$(docker create "${base_ref}")
# The wheel sits at / (the Dockerfile copies it before any WORKDIR).
wheel=$(docker run --rm --entrypoint sh "${base_ref}" -c 'ls /torch_tpu-*.whl')
docker cp "${cid}:${wheel}" .
mkdir -p assets
docker cp "${cid}:/torchtitan/assets/hf" assets/hf
docker rm "${cid}"

# The wheel is built against one torch nightly; install exactly that.
torch_version=$(docker run --rm "${base_ref}" python -c 'import importlib.metadata as m; print(m.version("torch"))')
python_version=$(docker run --rm "${base_ref}" python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "torch=${torch_version} python=${python_version} wheel=${wheel}"
echo "torch_version=${torch_version}" >> "${out}"
echo "python_version=${python_version}" >> "${out}"

# Free the disk for the build.
docker rmi "${base_ref}"
