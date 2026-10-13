"""Keep the dispatcher's activation dtype around upstream bf16 expert compute."""

from dataclasses import dataclass

from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts


class TpuGptOssGroupedExperts(GptOssGroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(GptOssGroupedExperts.Config):
        pass

    def forward(self, x, num_tokens_per_expert):
        # Upstream forces expert matmuls to bf16. The standard dispatcher's
        # scatter destination has x.dtype, including on the fp32 parity path.
        return super().forward(x, num_tokens_per_expert).type_as(x)
