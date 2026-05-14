"""JAX experiment job config."""

import dataclasses

from torchtitan.experiments.tpu.tpu_job_config import BaseTPUTrainerConfig


@dataclasses.dataclass
class JaxConfig:
    """Runtime knobs specific to the pure-JAX / Flax NNX trainer.
    """

    use_scan: bool = True
    tpu_megacore: bool = True
    tpu_num_slices: int = 1
    model_layer_override: int | None = None

    # When True, replace optax.adamw with a custom Adam that stores both mu
    # and nu in bf16 (upcast to fp32 per-leaf inside the update math). Useful
    # on fp32 master weights + memory-tight configs (e.g. Llama3 8B on v6e-4
    # where fp32 nu is 3.5 GiB/chip for the stacked MLP kernel alone). Off
    # by default; stock optax adamw is used when False.
    adamw_bf16_state: bool = False


@dataclasses.dataclass(kw_only=True, slots=True)
class JaxJobConfig(BaseTPUTrainerConfig):
    jax_config: JaxConfig = dataclasses.field(default_factory=JaxConfig)
