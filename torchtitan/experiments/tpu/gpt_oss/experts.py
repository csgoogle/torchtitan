"""Keep the dispatcher's activation dtype around upstream bf16 expert compute."""

from dataclasses import dataclass

import torch

from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts, GptOssMoE


class TpuGptOssGroupedExperts(GptOssGroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(GptOssGroupedExperts.Config):
        pass

    def forward(self, x, num_tokens_per_expert):
        # Upstream forces expert matmuls to bf16. The standard dispatcher's
        # scatter destination has x.dtype, including on the fp32 parity path.
        return super().forward(x, num_tokens_per_expert).type_as(x)


class _CombineBarrier(torch.autograd.Function):
    """Identity with an explicit derivative for the TPU scheduling barrier."""

    @staticmethod
    def forward(ctx, x):
        return torch.tpu.ops.optimization_barrier([x])[0]

    @staticmethod
    def backward(ctx, grad):
        return grad


class TpuGptOssMoE(GptOssMoE):
    """Keep expert-combine rounding separate from the block's residual add.

    On v6e, XLA fuses these across the module boundary and changes bf16
    rounding, even with excess precision disabled. A scheduling barrier
    preserves the eager combine result without changing values or gradients.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GptOssMoE.Config):
        pass

    def forward(self, x, **kwargs):
        output = super().forward(x, **kwargs)
        if output.device.type == "tpu":
            output = _CombineBarrier.apply(output)
        return output
