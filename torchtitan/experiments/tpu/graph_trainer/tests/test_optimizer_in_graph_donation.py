"""The in-graph optimizer must donate its parameter and moment inputs.

Without donation the step program takes the sharded fp32 parameters and both
AdamW moments as inputs and returns fresh copies as outputs, and XLA keeps both
sets resident for the whole step: 35.9 GiB in + 35.4 GiB out per TensorCore on
llama3-405B at 128-way FSDP, more than the whole activation stack the scan
offloads to host. With ``jax.buffer_donor`` on those inputs XLA aliases each to
its output (``input_output_alias``) and the second copy disappears.

Three things have to hold, each checked here on its own:

1. ``donated_input_indices`` picks exactly the optimizer-owned parameters and
   the ``_optstate_*_exp_avg`` / ``_exp_avg_sq`` buffers, in graph-input index
   space (a DTensor parameter widens to its local tensors).
2. The indices reach torch_tpu: the compiled StableHLO carries
   ``jax.buffer_donor = true`` on those arguments and no others.
3. The runtime contract the trainer now relies on: a donated input is consumed
   by the execution, so the module's parameter and moment entries must be
   replaced by the outputs (``rebind_model_and_optstate_buffers``) rather than
   written back with ``copy_``, and the next step runs on the rebound tensors.
"""

import re
import unittest

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.graph_trainer.make_fx_tracer import SubclassLayout
from torchtitan.experiments.tpu.compiled_optimizers.functional import (
    rebind_model_and_optstate_buffers,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    donated_input_indices,
    tpu_backend_compile_pass,
)


def _tpu_available() -> bool:
    try:
        return torch.tpu.device_count() > 0
    except Exception:  # torch_tpu not loaded, or no chip visible
        return False


# --- 1. index selection -----------------------------------------------------


class DonatedInputIndicesTest(unittest.TestCase):
    """Mirrors the naming convention of ``_register_optimizer_buffers``."""

    def test_selects_owned_params_and_both_moments_only(self):
        state_fqns = [
            "tok_embeddings.weight",  # 0: owned (its moments are 1, 2)
            "tok_embeddings._optstate_weight_exp_avg",  # 1: moment
            "tok_embeddings._optstate_weight_exp_avg_sq",  # 2: moment
            "layers.w1.weight",  # 3: owned (its moments are 4, 5)
            "layers.w1._optstate_weight_exp_avg",  # 4: moment
            "layers.w1._optstate_weight_exp_avg_sq",  # 5: moment
            "layers.frozen.weight",  # 6: requires_grad=False -> no moments -> keep
            "norm.scale",  # 7: plain buffer -> keep
            "_optstate_step",  # 8: scalar state, copied back, never donated
            "_optstate_lr",  # 9: scalar state, refilled each step, never donated
        ]
        self.assertEqual(donated_input_indices(state_fqns, {}), [0, 1, 2, 3, 4, 5])

    def test_root_level_param_uses_root_level_moments(self):
        state_fqns = ["weight", "_optstate_weight_exp_avg", "_optstate_weight_exp_avg_sq", "_optstate_step"]
        self.assertEqual(donated_input_indices(state_fqns, {}), [0, 1, 2])

    def test_dtensor_state_widens_to_its_local_tensors(self):
        state_fqns = [
            "a.weight",  # 0 -> graph inputs 0,1 (a 2-tensor subclass)
            "a._optstate_weight_exp_avg",  # 1 -> graph input 2
            "a._optstate_weight_exp_avg_sq",  # 2 -> graph input 3
            "b.bias",  # 3 -> graph input 4, not owned
            "b._optstate_bias_exp_avg",  # 4 -> graph input 5
            "b._optstate_bias_exp_avg_sq",  # 5 -> graph inputs 6,7
        ]
        layouts = {0: SubclassLayout(2, None), 5: SubclassLayout(2, None)}
        # b.bias IS owned (its exp_avg sibling exists) -> 4 is donated too.
        self.assertEqual(
            donated_input_indices(state_fqns, layouts), [0, 1, 2, 3, 4, 5, 6, 7]
        )

    def test_nothing_to_donate_without_in_graph_optimizer_state(self):
        self.assertEqual(donated_input_indices(["layers.0.w", "norm.scale"], {}), [])


# --- 3a. rebind semantics (device independent) -----------------------------


class _Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.register_buffer("_optstate_weight_exp_avg", torch.zeros(4))
        self.register_buffer("_optstate_weight_exp_avg_sq", torch.zeros(4))


class RebindTest(unittest.TestCase):
    def test_rebind_replaces_entries_with_the_outputs_without_copy(self):
        model = nn.Module()
        model.leaf = _Leaf()
        new_p, new_ea, new_eas = torch.ones(4), torch.full((4,), 2.0), torch.full((4,), 3.0)

        rebind_model_and_optstate_buffers(
            model, ["leaf.weight"], [new_p], [new_ea], [new_eas]
        )

        # The module now hands out a Parameter backed by the output's storage:
        # no copy, and the state extraction the next step performs sees it.
        self.assertIsInstance(model.leaf.weight, nn.Parameter)
        self.assertEqual(model.leaf.weight.data_ptr(), new_p.data_ptr())
        self.assertEqual(model.leaf._optstate_weight_exp_avg.data_ptr(), new_ea.data_ptr())
        self.assertEqual(model.leaf._optstate_weight_exp_avg_sq.data_ptr(), new_eas.data_ptr())
        self.assertEqual(dict(model.named_parameters())["leaf.weight"].data_ptr(), new_p.data_ptr())
        self.assertEqual(dict(model.named_buffers())["leaf._optstate_weight_exp_avg"].data_ptr(), new_ea.data_ptr())
        # Still a leaf that requires grad, as an optimizer-owned parameter must.
        self.assertTrue(model.leaf.weight.requires_grad)
        self.assertTrue(model.leaf.weight.is_leaf)

    def test_tied_parameters_stay_tied(self):
        model = nn.Module()
        model.leaf = _Leaf()
        model.head = nn.Module()
        model.head.weight = model.leaf.weight  # tied: one Parameter, two FQNs
        new_p = torch.ones(4)

        rebind_model_and_optstate_buffers(
            model,
            ["leaf.weight"],
            [new_p],
            [torch.zeros(4)],
            [torch.zeros(4)],
            fqn_to_canonical={"leaf.weight": "leaf.weight", "head.weight": "leaf.weight"},
        )

        self.assertIs(model.head.weight, model.leaf.weight)
        self.assertEqual(model.head.weight.data_ptr(), new_p.data_ptr())


# --- 2 + 3b. the torch_tpu contract, on a chip ------------------------------


def _adamw_like(p, m, g):
    """A stand-in for one in-graph optimizer update: returns fresh p and m."""
    new_m = m * 0.9 + g * 0.1
    new_p = p - new_m * 0.5
    return new_p, new_m


@unittest.skipUnless(_tpu_available(), "needs a TPU device")
class DonationOnDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # torch_tpu's compile passes read absl flags; the trainer runs under
        # absl.app, a pytest process does not, so mark them parsed here.
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["test"])

    def _graph(self):
        cpu = [torch.randn(8, 16) for _ in range(3)]
        gm = make_fx(_adamw_like)(*cpu)
        return gm, cpu

    def test_donated_arguments_are_marked_in_stablehlo(self):
        from torch_tpu._internal.compile.compiler import StaticCompiler

        gm, cpu = self._graph()
        tpu = [t.to("tpu") for t in cpu]
        executable = StaticCompiler(debug=True)(gm, tpu, donated_inputs=[0, 1])

        donors = re.findall(r"%arg(\d+): tensor<[^>]*> \{[^}]*jax\.buffer_donor = true", executable.mlir_text)
        self.assertEqual(sorted(int(d) for d in donors), [0, 1])
        self.assertEqual(executable.mlir_text.count("jax.buffer_donor"), 2)

    def test_compile_pass_donates_and_rebound_step_stays_correct(self):
        gm, cpu = self._graph()
        p, m, g = (t.to("tpu") for t in cpu)
        p_in, m_in = p, m

        gm = tpu_backend_compile_pass(gm, (p, m, g), donated_input_positions=(0, 1))
        new_p, new_m = gm(p, m, g)

        exp_p, exp_m = _adamw_like(*cpu)
        torch.testing.assert_close(new_p.cpu(), exp_p)
        torch.testing.assert_close(new_m.cpu(), exp_m)
        # The donated inputs are gone; the non-donated one is untouched.
        with self.assertRaisesRegex(RuntimeError, "deleted or donated"):
            p_in.cpu()
        with self.assertRaisesRegex(RuntimeError, "deleted or donated"):
            m_in.cpu()
        torch.testing.assert_close(g.cpu(), cpu[2])

        # What the trainer does between steps: rebind, then run on the rebound tensors.
        model = nn.Module()
        model.leaf = _Leaf()
        model.leaf.weight = nn.Parameter(p_in, requires_grad=False)
        model.leaf._optstate_weight_exp_avg = m_in
        model.leaf._optstate_weight_exp_avg_sq = torch.zeros_like(g)
        rebind_model_and_optstate_buffers(
            model, ["leaf.weight"], [new_p], [new_m], [torch.zeros_like(new_m)]
        )
        self.assertEqual(model.leaf.weight.data_ptr(), new_p.data_ptr())

        new_p2, new_m2 = gm(model.leaf.weight, model.leaf._optstate_weight_exp_avg, g)
        exp_p2, exp_m2 = _adamw_like(exp_p, exp_m, cpu[2])
        torch.testing.assert_close(new_p2.cpu(), exp_p2)
        torch.testing.assert_close(new_m2.cpu(), exp_m2)


if __name__ == "__main__":
    unittest.main()
