# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real-arithmetic rewrite of `ComplexRoPE.apply_rotary_emb`.

Required by scan-over-layers. `ComplexRoPE.apply_rotary_emb` does

    xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * rope_cache).flatten(3)

Inside a `scan` combine_fn the HOP re-traces the body functionally, where
`view_as_complex` becomes `view_as_complex_copy` — which has no autograd
formula:

    NotImplementedError: view_as_complex_copy does not support automatic
    differentiation for outputs with complex dtype

`query` requires grad, so this fires on every attention layer and the whole
scan fails to trace. The *cache* is a non-persistent buffer and never requires
grad, so it is left complex and only read through `view_as_real`; the only op
removed is the differentiable complex view.

This is a mechanical rewrite of the same complex multiply, not a change of
convention: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, with the same
adjacent-dim pairing and the same interleaved output layout. PyTorch's complex
multiply evaluates exactly that expression, so the two forms agree bit-for-bit
in fp32; `tests/test_rope_real_arithmetic.py` asserts it, and the scan bring-up
harness re-checks it end-to-end on device before any scan number is reported.

Safe to apply unconditionally — it is a pure refactor of one static method.
"""

import torch

from torchtitan.tools.logging import logger

_APPLIED = False


def _apply_rotary_emb_real(
    query: torch.Tensor,
    key: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """`ComplexRoPE.apply_rotary_emb` without a differentiable complex view.

  Args:
    query: `(batch, seq_len, n_heads, head_dim)`.
    key: `(batch, seq_len, n_kv_heads, head_dim)`.
    rope_cache: complex cache broadcastable to `(..., head_dim // 2)`.

  Returns:
    Rotated query and key, same shapes and dtypes as the inputs.
  """
  # `rope_cache` is a buffer view: complex, no grad. `view_as_real` on it is
  # not differentiated, so it does not hit the HOP limitation.
  cache = torch.view_as_real(rope_cache)
  cos = cache[..., 0]
  sin = cache[..., 1]

  def rotate(x: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        [real * cos - imag * sin, real * sin + imag * cos], dim=-1
    )
    return rotated.flatten(3).type_as(x)

  return rotate(query), rotate(key)


def use_real_arithmetic_rope_patch() -> bool:
  """Replaces `ComplexRoPE.apply_rotary_emb` in place. Idempotent."""
  global _APPLIED
  if _APPLIED:
    return True

  from torchtitan.models.common import rope as rope_module  # pylint: disable=g-import-not-at-top

  rope_module.ComplexRoPE.apply_rotary_emb = staticmethod(_apply_rotary_emb_real)
  _APPLIED = True
  logger.info(
      "Patched ComplexRoPE.apply_rotary_emb with the real-arithmetic form"
      " (required for scan-over-layers; bit-equivalent to the complex form)."
  )
  return True
