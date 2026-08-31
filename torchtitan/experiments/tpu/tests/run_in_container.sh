#!/usr/bin/env bash
# Runs one test module inside a torch_tpu container against a real TPU.
#
# The device modules need a chip and a torch_tpu build; the CPU-only ones do
# not, though running them here too is the only way to get the image's torch
# version rather than the host conda env's. This wrapper exists because
# torch_tpu's PjRt client will not initialize without a slice topology in the
# environment, and working that out per-host is the part that wastes time.
#
# NPROC is the tier: 1 for the single-chip modules, 8 for
# test_tpu_graph_trainer_distributed, which wants the whole host.
#
# Usage:
#   IMG=<torch_tpu container image> ./run_in_container.sh <test-module> [args...]
#
# Env:
#   IMG    required -- container image carrying torch_tpu and its dependencies.
#   WHEEL  path to a torch_tpu wheel to install over the image's copy. Empty
#          (the default) uses whatever the image ships, which is what you want
#          unless you are testing a patched torch_tpu.
#   NPROC  devices to claim (default 1).
#   TT     torchtitan checkout to mount (default: the one containing this file).
#   ENVS   space-separated KEY=VALUE pairs passed through to the container.
#
# Example:
#   IMG=my-registry/torchtitan-tpu:latest \
#     ./run_in_container.sh torchtitan.experiments.tpu.tests.test_scan_lowering

set -uo pipefail

IMG="${IMG:?set IMG to a container image carrying torch_tpu}"
WHEEL="${WHEEL:-}"
NPROC="${NPROC:-1}"
TT="${TT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
MODULE="${1:?usage: run_in_container.sh <test-module> [args...]}"
shift

WHEEL_MOUNT=()
# No apostrophe: this string is interpolated into a single-quoted context
# inside the `bash -c` script below, where one would terminate the quote and
# make the generated script a syntax error.
WHEEL_INSTALL="echo 'using the torch_tpu that ships in the image'"
if [[ -n "$WHEEL" ]]; then
  WHEEL_MOUNT=(-v "$(dirname "$WHEEL")":/whl:ro)
  WHEEL_INSTALL="pip install -q --no-deps --force-reinstall /whl/$(basename "$WHEEL")"
fi

ENV_ARGS=()
for kv in ${ENVS:-}; do ENV_ARGS+=(-e "$kv"); done

# core=0: torch_tpu aborts via LOG(FATAL), and the default handler then tries to
# dump a multi-GB core inside the container -- slow, and it fills the host disk.
exec docker run --rm --privileged --net=host \
  --shm-size=32g --ulimit nofile=100000:100000 --ulimit core=0 \
  -v "$TT":/torchtitan \
  "${WHEEL_MOUNT[@]}" "${ENV_ARGS[@]}" \
  --entrypoint bash "$IMG" -c "
set -uo pipefail
$WHEEL_INSTALL
cd /torchtitan

# torch_tpu's PjRt client refuses to initialize without
# TORCH_TPU_SLICEBUILDER_ADDRESSES + TORCH_TPU_TOPOLOGY in the environment.
# prepare_tpu_environment is what the single-host launcher uses to derive them;
# it sets them on its own process, so re-export them into this shell.
eval \"\$(python -c '
import os
from torch_tpu._internal.distributed.launchers.singlehost_wrapper import prepare_tpu_environment
prepare_tpu_environment($NPROC)
for k in (\"TORCH_TPU_SLICEBUILDER_ADDRESSES\", \"TORCH_TPU_TOPOLOGY\", \"TORCH_TPU_XPROF_SESSION_ID\"):
  print(f\"export {k}={os.environ[k]}\")
')\"

python -m $MODULE $*
"
