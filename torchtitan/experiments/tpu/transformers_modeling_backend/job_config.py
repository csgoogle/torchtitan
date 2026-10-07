"""Job config for the HF backend on the TPU lane."""

import dataclasses

from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


@dataclasses.dataclass(kw_only=True, slots=True)
class TPUTransformersBackendConfig(TPUTrainerConfig):
  """A TPU job config that also carries the HuggingFace model id.

  The HF backend reads the id off the *job* config -- ``HFTransformerModel.
  Config.update_from_config`` does ``getattr(config, "hf_model", "")`` -- so it
  has to be a top-level field rather than one of the per-model sub-configs.
  ``TPUTrainerConfig`` is a slotted dataclass, so it cannot simply be set at
  runtime; subclassing is what adds the slot.

  Subclassing ``TPUTrainerConfig`` (rather than the backend's
  ``SFTTrainer.Config``) is what lets the TPU entry point accept this config
  unchanged: ``gmain`` reaches for ``config.tpu_config``, and ``train.py``
  gates its TPU behavior on ``isinstance(config, TPUTrainerConfig)``.
  """

  hf_model: str = ""
