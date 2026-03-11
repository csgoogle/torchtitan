from dataclasses import dataclass, field
from torch import nn
from torchtitan.config import JobConfig
from torchtitan.models.flux.model.args import FluxModelArgs as NativeFluxModelArgs
from torchtitan.models.flux.model.autoencoder import AutoEncoderParams
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger

@dataclass
class FluxModelArgs(NativeFluxModelArgs):
    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        # Lazy import to avoid circular dependencies.
        from torchtitan.experiments.tpu.tpu_job_config import TPUJobConfig
        # Check if we are running a TPU Job
        if isinstance(job_config, TPUJobConfig):
            # Add TPU specific updates here if needed
            logger.info("TPUJobConfig detected for Flux.")
            pass

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return super().get_nparams_and_flops(model, seq_len)
