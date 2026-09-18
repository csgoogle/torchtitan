"""Meta-device init that does not destroy buffer values.

torchtitan builds every model under `torch.device("meta")` and then calls
`to_empty()` + `init_weights()`. That is correct for torchtitan's own models,
whose state is entirely parameters, but it silently corrupts HuggingFace
models, whose `__init__` computes real *buffers*:

    GPT-2      causal-mask bias
    Gemma      rope inv_freq, embedding normalizer
    OPT/many   position offsets

A meta build never runs those computations, `to_empty()` hands back
uninitialized memory, and `_init_weights()` only touches parameters -- so the
buffers stay garbage. The symptoms are not obviously related to each other:
`Cannot copy out of meta tensor`, silent NaN losses, or a loss an order of
magnitude too high. torchtitan patched exactly one instance of this by
recomputing rope's `inv_freq` by hand; every other buffer is still wrong.

The fix is to stop treating parameters and buffers alike:

  * `params_meta_buffers_real()` runs `__init__` on a real device so buffers
    are actually computed, and swaps each parameter to meta as it is
    registered. Weight init then lands on a meta tensor and is a no-op, so
    this keeps the speed that made meta init worth using (a CPU build of two
    Qwen3-9B layers costs 81 s; this does not pay that).
  * `to_empty_keep_buffers()` re-allocates parameters on the device but
    *moves* buffers, preserving what `__init__` computed.
"""

import contextlib

import torch
from torch import nn


@contextlib.contextmanager
def params_meta_buffers_real(build_device: str = "cpu"):
    """Build a model with parameters on meta and buffers really computed."""
    original_register = nn.Module.register_parameter

    def register_parameter(module, name, param):
        original_register(module, name, param)
        registered = module._parameters.get(name)
        if registered is not None and not registered.is_meta:
            module._parameters[name] = nn.Parameter(
                registered.to("meta"), requires_grad=registered.requires_grad
            )

    nn.Module.register_parameter = register_parameter
    try:
        # Escape any enclosing meta-device context so buffer math is real.
        with torch.device(build_device):
            yield
    finally:
        nn.Module.register_parameter = original_register


def to_empty_keep_buffers(model: nn.Module, device) -> nn.Module:
    """`to_empty()` for parameters; a plain move for buffers.

    torch's `to_empty()` calls `torch.empty_like` on every tensor, buffers
    included, which throws away exactly the values this module exists to
    protect.
    """
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            module._parameters[name] = nn.Parameter(
                torch.empty_like(param, device=device),
                requires_grad=param.requires_grad,
            )
        for name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            module._buffers[name] = (
                torch.empty_like(buf, device=device) if buf.is_meta else buf.to(device)
            )
    return model
