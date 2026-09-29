"""`pallas_grouped_mm_w2_tgmm_tiling` reaches the W2 dWeight and nothing else.

One ``pallas_grouped_mm_tgmm_tiling`` tuple used to feed two kernels with
different contracted axes: the paired W1/W3 dWeight (K = ``dim``) and the
unpaired W2 dWeight (K = ``hidden_dim``). A K tile good for the first can be
too large for the second, so W2 gets its own. These tests pin the routing;
the kernels themselves are patched out, so this runs on CPU.
"""

from unittest import mock

import torch
from absl.testing import absltest

from torchtitan.experiments.tpu import grouped_experts as ge
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

DIM, HIDDEN, EXPERTS, ROWS = 8, 16, 2, 6
SHARED = (8, 16, 0)
W2_OWN = (4, 8, 4)  # Any N tile: the full-N rule is the paired kernel's alone.


def _experts(**overrides) -> ge.TpuGroupedExperts:
  kwargs = dict(
      dim=DIM,
      hidden_dim=HIDDEN,
      num_experts=EXPERTS,
      token_dispatcher=LocalTokenDispatcher.Config(
          num_experts=EXPERTS, top_k=1
      ),
      use_pallas_grouped_mm=True,
      pallas_grouped_mm_tgmm_tiling=SHARED,
  )
  kwargs.update(overrides)
  return ge.TpuGroupedExperts(ge.TpuGroupedExperts.Config(**kwargs))


def _fake_segment(x, w, counts, **kwargs):
  del counts, kwargs
  return x.new_zeros(x.shape[0], w.shape[-1])


def _fake_paired(x, w1, w3, counts, **kwargs):
  del w3, counts, kwargs
  out = x.new_zeros(x.shape[0], w1.shape[-1])
  return out, out


class W2TilingTest(absltest.TestCase):

  def _run(self, experts):
    x = torch.zeros(ROWS, DIM)
    counts = torch.tensor([ROWS // 2, ROWS - ROWS // 2])
    with mock.patch.object(
        ge, "segment_matmul_pallas_2d", side_effect=_fake_segment
    ) as segment, mock.patch.object(
        ge, "paired_segment_matmul_pallas_2d", side_effect=_fake_paired
    ) as paired:
      experts._experts_forward(x, counts)
    return segment, paired

  def _w2_call(self, segment):
    # W2 is always the last unpaired call: gate/up come first when unpaired.
    return segment.call_args_list[-1]

  def test_default_follows_the_shared_tiling(self):
    segment, paired = self._run(_experts(use_pallas_paired_grouped_mm=True))
    self.assertEqual(paired.call_count, 1)
    self.assertEqual(segment.call_count, 1)
    self.assertEqual(self._w2_call(segment).kwargs["tgmm_tiling"], SHARED)

  def test_paired_path_routes_the_w2_tiling_to_w2_only(self):
    segment, paired = self._run(
        _experts(
            use_pallas_paired_grouped_mm=True,
            pallas_grouped_mm_w2_tgmm_tiling=W2_OWN,
        )
    )
    # The paired W1/W3 dWeight still takes its K tile from the shared tuple.
    kwargs = paired.call_args.kwargs
    self.assertEqual(kwargs["tgmm_tile_m"], SHARED[0])
    self.assertEqual(kwargs["tgmm_tile_k"], SHARED[1])
    self.assertEqual(self._w2_call(segment).kwargs["tgmm_tiling"], W2_OWN)

  def test_unpaired_path_keeps_gate_and_up_on_the_shared_tiling(self):
    segment, paired = self._run(
        _experts(
            use_pallas_paired_grouped_mm=False,
            pallas_grouped_mm_w2_tgmm_tiling=W2_OWN,
        )
    )
    self.assertEqual(paired.call_count, 0)
    self.assertEqual(segment.call_count, 3)
    gate, up, w2 = segment.call_args_list
    self.assertEqual(gate.kwargs["tgmm_tiling"], SHARED)
    self.assertEqual(up.kwargs["tgmm_tiling"], SHARED)
    self.assertEqual(w2.kwargs["tgmm_tiling"], W2_OWN)

  def test_full_n_rule_applies_to_the_paired_tuple_not_w2(self):
    # A non-zero N tile on W2 is fine; on the shared tuple it is still refused.
    _experts(
        use_pallas_paired_grouped_mm=True,
        pallas_grouped_mm_w2_tgmm_tiling=W2_OWN,
    )
    with self.assertRaises(ValueError):
      _experts(
          use_pallas_paired_grouped_mm=True,
          pallas_grouped_mm_tgmm_tiling=(8, 16, 4),
      )


if __name__ == "__main__":
  absltest.main()
