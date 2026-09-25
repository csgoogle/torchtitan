"""TorchTPU lane for the HuggingFace transformers modeling backend.

Runs `torchtitan/experiments/transformers_modeling_backend` on TorchTPU with
SimpleFSDP, reusing that backend's model, its inputs, and the TPU lane's
existing trainer unchanged. Only two things differ from the GPU path:

- the job config is a `TPUTrainerConfig`, so the TPU entry point
  (`experiments/tpu/train.py`) accepts it, and
- data parallelism is SimpleFSDP rather than FSDP2 `fully_shard`.

Launch:

    python -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.transformers_modeling_backend \
        --config=hf_debugmodel
"""

from .job_config import TPUTransformersBackendConfig
from .parallelize import parallelize_hf_transformers_tpu

__all__ = ["TPUTransformersBackendConfig", "parallelize_hf_transformers_tpu"]
