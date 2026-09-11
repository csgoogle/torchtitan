# Import torch before any test module runs. On TPU hosts torch's backend
# autoload imports torch_tpu -> dynamo -> FSDP -> spmd_types; a test that
# imports spmd_types before torch would otherwise re-enter a half-initialised
# spmd_types and torch fails to load. See torchtitan/__init__.py.
import torch  # noqa: F401  # isort: skip

