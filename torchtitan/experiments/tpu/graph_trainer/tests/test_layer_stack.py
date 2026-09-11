# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stacking a decoder's layer weights must change storage and nothing else.

Every check here is against an unstacked model built the same way, because the
failure modes are silent: a stack initialized in the wrong order trains to a
plausible loss, and a dropped view shows up only as a checkpoint that no longer
loads.
"""

import copy
import dataclasses
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch._functorch import config as functorch_config, partitioners
from torch.distributed.tensor import DTensor, Shard

from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.layer_stack import STACK_ATTR
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import gather_slice
from torchtitan.experiments.tpu.workarounds import use_real_rope_arithmetic_patch
from torchtitan.experiments.tpu.llama3 import model_registry
from torchtitan.experiments.tpu.qwen3 import model_registry as qwen3_registry
from torchtitan.models.common.rope import ComplexRoPE


SEQ_LEN = 32
SEED = 1234


def _config(registry=model_registry):
    """A debug model at ``SEQ_LEN`` so the RoPE cache stays small.

    The TPU registries install ``SplashAttention``, whose reference path runs
    on CPU; upstream's flex inner attention has no CPU kernel.
    """
    return registry("debugmodel", seq_len=SEQ_LEN).model


def _tokens(model, batch: int = 2):
    """A token-major batch of ``batch`` rows and its per-row positions."""
    tokens = torch.randint(0, model.config.vocab_size, (batch * SEQ_LEN,))
    positions = torch.arange(SEQ_LEN).repeat(batch)
    return tokens, positions


def _build(
    *,
    stacked: bool,
    scan: bool = False,
    param_dtype: str = "float32",
    config=None,
):
    """Build a model the way the trainer does: parallelize, to_empty, init."""
    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
    )
    training = TrainingConfig(
        mixed_precision_param=param_dtype, mixed_precision_reduce="float32"
    )
    with torch.device("meta"):
        model = (config or _config()).build()
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
        stack_layers=stacked,
        scan_layers=scan,
    )
    model.to_empty(device="cpu")
    torch.manual_seed(SEED)
    model.init_weights()
    return model


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestLayerStack(unittest.TestCase):
    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12377",
                world_size=1,
                rank=0,
            )
        # `scan` re-traces its body through functionalization, which has no
        # derivative for `view_as_complex_copy`. On a real run the TPU lane
        # installs this before the first forward; see `workarounds`.
        unpatched = ComplexRoPE.__dict__["apply_rotary_emb"]
        use_real_rope_arithmetic_patch()
        self.addCleanup(setattr, ComplexRoPE, "apply_rotary_emb", unpatched)

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_state_dict_is_identical(self):
        """Checkpoints stay in the unrolled format, down to the bits.

        This is the check that an existing 70B checkpoint still loads.
        """
        plain = _build(stacked=False).state_dict()
        stacked = _build(stacked=True).state_dict()

        self.assertEqual(set(stacked), set(plain))
        for key, want in plain.items():
            torch.testing.assert_close(
                stacked[key].to_local(), want.to_local(), rtol=0, atol=0, msg=key
            )

    def test_init_differs_per_row(self):
        """Every layer's row is different.

        llama3 scales each layer's init by its depth. Initializing the stack in
        the wrong order -- or initializing one row and broadcasting it -- gives a
        model that still trains, so compare the rows against each other.
        """
        model = _build(stacked=True)
        state = model.state_dict()
        rows = [
            state[f"layers.{i}.attention.wo.weight"] for i in range(len(model.layers))
        ]
        for i in range(1, len(rows)):
            self.assertFalse(
                torch.equal(rows[0].to_local(), rows[i].to_local()),
                f"layer {i} was initialized identically to layer 0",
            )

    def test_module_tree_is_unchanged(self):
        """All n layer modules stay registered, so tree walkers still find them.

        The splash-attention patcher matches on class name over
        ``model.modules()``, activation checkpointing parses layer numbers out
        of FQNs, and the device move visits every module. None of them are
        touched by this change, and this is why.
        """
        plain = dict(_build(stacked=False).named_modules())
        stacked = dict(_build(stacked=True).named_modules())

        self.assertEqual(set(stacked) - set(plain), {STACK_ATTR})
        self.assertEqual(set(plain) - set(stacked), set())

    def test_each_weight_is_a_parameter_exactly_once(self):
        """Otherwise the optimizer applies every layer gradient twice.

        The row views exist only while ``init_states`` runs. If they are left
        registered, ``model.parameters()`` yields each weight both as a row and
        inside its stack.
        """
        model = _build(stacked=True)
        params = list(model.parameters())
        self.assertEqual(len({id(p) for p in params}), len(params))

        # The layer modules keep no parameters of their own, and the holder holds
        # nothing but the stacks -- one per parameter of a single layer.
        self.assertEqual(list(model.layers.parameters()), [])
        holder = getattr(model, STACK_ATTR)
        plain_layer = dict(_build(stacked=False).layers["0"].named_parameters())
        self.assertEqual(
            len(dict(holder.named_parameters(recurse=False))), len(plain_layer)
        )
        self.assertEqual(
            len(list(holder.parameters(recurse=False))), len(list(holder.parameters()))
        )

    def test_stacks_are_sharded_off_the_layer_axis(self):
        """Sharding dim 0 would split one layer's weight across ranks.

        It would also silently fall back to ``Replicate`` whenever ``n_layers``
        is below the mesh size, which is the usual case -- so pin the dim.
        """
        holder = getattr(_build(stacked=True), STACK_ATTR)
        for name, stack in holder.named_parameters(recurse=False):
            self.assertEqual(stack.placements, (Shard(1),), name)

    def test_forward_matches_unrolled(self):
        model = _build(stacked=False)
        stacked = _build(stacked=True)
        tokens, positions = _tokens(model)

        torch.testing.assert_close(
            stacked(tokens, positions), model(tokens, positions), rtol=0, atol=0
        )

    def test_scan_forward_matches_the_loop(self):
        plain = _build(stacked=False)
        scanned = _build(stacked=True, scan=True)
        tokens, positions = _tokens(plain)

        torch.testing.assert_close(
            scanned(tokens, positions), plain(tokens, positions), rtol=1e-6, atol=1e-6
        )

    def test_the_body_recomputes_its_gather_in_backward(self):
        """The in-body gather is worth nothing unless backward redoes it.

        Left at the default budget the partitioner prices a collective as
        expensive to redo and saves the gathered per-layer weight across the
        cut, so the gather is inside the loop but its result is carried out of
        it. ``scan_loop`` opens an ``activation_memory_budget=0.0`` window to
        stop that; this pins both that the window reaches the body's partition
        and that it closes again -- the config is process-global.
        """
        captured = []
        partition = partitioners.min_cut_rematerialization_partition

        def capture(joint, inputs, **kwargs):
            forward, backward = partition(joint, inputs, **kwargs)
            captured.append((functorch_config.activation_memory_budget, backward))
            return forward, backward

        model = _build(stacked=True, scan=True)
        tokens, positions = _tokens(model)
        default = functorch_config.activation_memory_budget
        with patch.object(
            partitioners, "min_cut_rematerialization_partition", capture
        ):
            model(tokens, positions).sum().backward()

        self.assertTrue(captured, "the scan body was never partitioned")
        budget, backward = captured[0]
        self.assertEqual(budget, 0.0)
        self.assertTrue(
            any("all_gather" in str(n.target) for n in backward.graph.nodes),
            "the body's backward reuses a saved gather instead of redoing it",
        )
        self.assertEqual(functorch_config.activation_memory_budget, default)

    def test_scan_gradients_match_the_loop(self):
        """Per layer, because a severed carry chain keeps the loss exact.

        If the scan's init carry does not carry a gradient, the inter-layer
        chain is cut: the loss is unchanged, no error is raised, and only the
        layer at one end of the stack is visibly wrong.
        """
        plain = _build(stacked=False)
        scanned = _build(stacked=True, scan=True)
        tokens, positions = _tokens(plain)

        for model in (plain, scanned):
            model(tokens, positions).sum().backward()

        # `_parameters`, not `get_parameter`: SimpleFSDP shadows each name with a
        # property returning the gathered tensor, which has no `.grad`.
        stacks = getattr(scanned, STACK_ATTR)._parameters
        for i, layer in enumerate(plain.layers.values()):
            for name, param in layer.named_parameters():
                stack = stacks[name.replace(".", "__")]
                torch.testing.assert_close(
                    stack.grad[i].to_local(),
                    param.grad.to_local(),
                    rtol=1e-5,
                    atol=1e-6,
                    msg=f"layers.{i}.{name}",
                )

    def test_the_body_gathers_in_the_compute_dtype(self):
        """The gather is also the cast, exactly as SimpleFSDP's is.

        The stacks are stored in fp32 and the layers run in bf16; unrolled,
        SimpleFSDP's parametrization casts as it gathers. If the in-body gather
        skipped that, the scanned model would run its layers in fp32 -- slower,
        more memory, and numerically *closer* to the reference, so a loss
        comparison would not catch it.
        """
        model = _build(stacked=True, scan=True, param_dtype="bfloat16")
        # Through the installed container, so this is the plan the body will
        # use -- the body itself cannot be instrumented, as dynamo rejects any
        # side effect inside a HOP.
        stack = model.layers.stack
        self.assertTrue(stack.fsdp_plan, "no gather plan was installed")

        holder = getattr(model, STACK_ATTR)
        for name, spec in stack.fsdp_plan.items():
            param = holder._parameters[name]
            local = param.to_local() if isinstance(param, DTensor) else param
            gathered = gather_slice(local[0], spec)
            self.assertEqual(gathered.dtype, torch.bfloat16, name)
            # ... and the stored weight is untouched, so the optimizer still
            # steps in fp32.
            self.assertEqual(param.dtype, torch.float32, name)

    def test_the_optimizer_sees_each_stack_once(self):
        """Param groups are built from ``named_parameters``, which the stacks change.

        Under the default catch-all pattern the assignment must still be exactly
        the stacks plus the non-layer weights, each once. Compared by identity:
        the row views and the stacks share storage, so a name- or value-based
        check would not notice the optimizer stepping a view.
        """
        model = _build(stacked=True)
        optimizers = OptimizersContainer(default_adamw(), model_parts=[model])

        assigned = [
            p
            for opt in optimizers
            for group in opt.param_groups
            for p in group["params"]
        ]
        self.assertEqual(len({id(p) for p in assigned}), len(assigned))
        self.assertEqual(
            {id(p) for p in assigned}, {id(p) for p in model.parameters()}
        )
        holder = getattr(model, STACK_ATTR)
        self.assertTrue(
            {id(p) for p in holder.parameters()} <= {id(p) for p in assigned}
        )

    def test_loads_an_unrolled_checkpoint(self):
        """A checkpoint written by the unrolled model loads into the stacked one."""
        plain = _build(stacked=False)
        stacked = _build(stacked=True)
        tokens, positions = _tokens(plain)

        # Perturb the source so a no-op load cannot pass.
        with torch.no_grad():
            for p in plain.parameters():
                p.add_(torch.randn_like(p))
        checkpoint = copy.deepcopy(plain.state_dict())

        stacked.load_state_dict(checkpoint)
        torch.testing.assert_close(stacked(tokens, positions), plain(tokens, positions), rtol=0, atol=0)


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestQwen3(unittest.TestCase):
    """The same two properties on a second model, to pin that none of this is llama3.

    Stacking reads the layers' own parameters and never their names, so a model
    with a different attention shape, a different norm placement and its own
    RoPE should need no code. That is a claim about the implementation, and the
    only way to keep it true is to check it against something that is not
    llama3.
    """

    def setUp(self):
        # `scan` traces its body through dynamo. Tracing a *differently shaped*
        # body in the same process reuses the cached shape environment and
        # fails on an unbacked symbol; llama3's scan tests run first, so
        # without this the qwen3 ones fail only in a full-file run.
        torch._dynamo.reset()
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12378",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _build(self, **kwargs):
        # Weight tying off, and not because of stacking: `init_states` re-ties
        # by assigning `lm_head.weight` to `tok_embeddings.weight`, and under
        # SimpleFSDP that attribute is a parametrization returning a plain
        # gathered tensor, which `nn.Module.__setattr__` refuses. The unrolled
        # arm fails identically, so it is a gap in this lane, not in the stack.
        #
        # With tying off nothing initializes `tok_embeddings` -- `init_states`
        # was its only writer, and `skip_param_init` skips it -- so it holds
        # whatever `to_empty` left. Hence `_pair` below, and the restriction of
        # the state-dict comparison to the decoder.
        config = dataclasses.replace(
            _config(qwen3_registry), enable_weight_tying=False
        )
        return _build(config=config, **kwargs)

    def _pair(self, **kwargs):
        """A plain model and a stacked one holding exactly its weights."""
        plain = self._build(stacked=False)
        other = self._build(**kwargs)
        other.load_state_dict(copy.deepcopy(plain.state_dict()))
        return plain, other

    def test_layer_weights_are_identical(self):
        plain = self._build(stacked=False).state_dict()
        stacked = self._build(stacked=True).state_dict()

        self.assertEqual(set(stacked), set(plain))
        layer_keys = [k for k in plain if k.startswith("layers.")]
        self.assertTrue(layer_keys)
        for key in layer_keys:
            torch.testing.assert_close(
                stacked[key].to_local(),
                plain[key].to_local(),
                rtol=0,
                atol=0,
                msg=key,
            )

    def test_scan_forward_matches_the_loop(self):
        plain, scanned = self._pair(stacked=True, scan=True)
        tokens, positions = _tokens(plain)

        torch.testing.assert_close(
            scanned(tokens, positions), plain(tokens, positions), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
