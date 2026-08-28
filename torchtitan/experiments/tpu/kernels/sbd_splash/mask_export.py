# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""sbd fork: reconstruct the realized dropout keep-mask outside the attention
kernel (the axlearn get_dropout_mask counterpart, quantum-keyed).

Generates keep = (bits >= rate*2^32) per DROPOUT_QUANTUM tile with
pltpu.prng_seed(seed0, seed1, head, q_quantum, kv_quantum) -- the identical
hash the fwd kernel uses, so the exported mask reproduces the in-kernel draw
exactly (same chip/libtpu). Block-size-independent by construction.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from torchtitan.experiments.tpu.kernels.sbd_splash.splash_attention_kernel import DROPOUT_QUANTUM, _mix_dropout_seed


def _mask_kernel(seed_ref, out_ref, *, thresh, band_width, version, thresh8,
                 block_q_rows=None):
    h = pl.program_id(0)
    i = pl.program_id(1)
    j = pl.program_id(2)
    w0, w1 = _mix_dropout_seed(seed_ref[0], seed_ref[1], h, i, j)
    pltpu.prng_seed(w0, w1)
    if version == 5:
        # v5: one chain per (q-BLOCK, band); i is the q-block index here and
        # the grid uses block_q-sized rows (see dropout_keep_mask).
        bits5 = pltpu.prng_random_bits((block_q_rows, band_width))
        keep = lax.bitcast_convert_type(bits5, jnp.uint32) >= jnp.uint32(thresh)
        out_ref[...] = keep.astype(jnp.int8)
    elif version == 4:
        words = pltpu.prng_random_bits((DROPOUT_QUANTUM // 4, band_width))
        by = pltpu.bitcast(words, jnp.int8)  # sublane-expand (qt, band)
        _t8v = thresh8 ^ 0x80
        t8f = jnp.float32(_t8v - 256 if _t8v >= 128 else _t8v)
        xs = (by ^ jnp.int8(-128)).astype(jnp.float32)
        out_ref[...] = (xs >= t8f).astype(jnp.int8)
    elif version == 3:
        words = lax.bitcast_convert_type(
            pltpu.prng_random_bits((DROPOUT_QUANTUM, band_width // 4)),
            jnp.uint32,
        )
        lanes = [
            ((words >> (8 * l)) & jnp.uint32(0xFF)) >= jnp.uint32(thresh8)
            for l in range(4)
        ]
        keep = jnp.concatenate(
            [ln.astype(jnp.int8) for ln in lanes], axis=1
        )
        out_ref[...] = keep
    else:
        bits = pltpu.prng_random_bits((DROPOUT_QUANTUM, band_width))
        keep = lax.bitcast_convert_type(bits, jnp.uint32) >= jnp.uint32(thresh)
        out_ref[...] = keep.astype(jnp.int8)


@functools.partial(
    jax.jit,
    static_argnames=[
        "num_heads", "q_seq", "kv_seq", "rate", "interpret", "version",
        "band_width", "block_q",
    ],
)
def dropout_keep_mask(
    seed_words, *, num_heads, q_seq, kv_seq, rate, interpret=False,
    version=1, band_width=None, block_q=None,
):
    """Returns int8 [num_heads, q_seq, kv_seq]; 1 == KEEP.

    version=1: 256x256 quantum draws (j = kv quantum index).
    version=2: (256, band_width) draws (j = kv BAND index); band_width must
    equal the kernel config's block_kv_compute (== block_kv_dkv_compute).
    """
    if version == 6:
        # v019: orientation-free per-element counter hash -- pure jnp path,
        # deliberately INDEPENDENT of the kernel implementation (no Pallas):
        # the export-vs-kernel agreement is the identity gate's evidence.
        from torchtitan.experiments.tpu.kernels.sbd_splash.splash_attention_kernel import _hash_keep_bool

        w0 = jnp.asarray(seed_words).reshape(-1)[0].astype(jnp.int32)
        w1 = jnp.asarray(seed_words).reshape(-1)[1].astype(jnp.int32)
        q_abs = lax.broadcasted_iota(jnp.int32, (num_heads, q_seq, kv_seq), 1)
        kv_abs = lax.broadcasted_iota(jnp.int32, (num_heads, q_seq, kv_seq), 2)
        h_abs = lax.broadcasted_iota(jnp.int32, (num_heads, q_seq, kv_seq), 0)
        return _hash_keep_bool(w0, w1, h_abs, q_abs, kv_abs, rate).astype(
            jnp.int8
        )
    qt = DROPOUT_QUANTUM
    bw = qt if version == 1 else int(band_width)
    # v5 keys per (q-BLOCK, band): grid rows are block_q-sized, not quantum.
    rows = int(block_q) if version == 5 else qt
    assert q_seq % rows == 0 and kv_seq % bw == 0
    thresh = min(int(rate * (2**32)), 2**32 - 1)
    thresh8 = min(int(rate * 256), 255)
    return pl.pallas_call(
        functools.partial(_mask_kernel, thresh=thresh, band_width=bw,
                          version=version, thresh8=thresh8,
                          block_q_rows=rows),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[],
            out_specs=pl.BlockSpec(
                (None, rows, bw), lambda h, i, j, *_: (h, i, j)
            ),
            grid=(num_heads, q_seq // rows, kv_seq // bw),
        ),
        out_shape=jax.ShapeDtypeStruct((num_heads, q_seq, kv_seq), jnp.int8),
        interpret=interpret,
    )(seed_words)
