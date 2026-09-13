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
from torch import nn
from torch.distributed.tensor import Shard

from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.config import derive
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    check_layers_are_stackable,
    STACK_ATTR,
    with_tpu_stacked_layers,
)
from torchtitan.experiments.tpu.layer_loop import tpu_model_class
from torchtitan.models.llama3 import model_registry
from torchtitan.models.qwen3 import model_registry as qwen3_registry


SEQ_LEN = 32
SEED = 1234


def _layer_loop(config):
    """The unstacked arm: the same model class, without the stacking."""
    return derive(config, tpu_model_class(type(config)).Config)


def _config(registry=model_registry):
    """A debug model, shortened to ``SEQ_LEN`` so the RoPE cache stays small.

    Stock and unconverted; ``_build`` applies the rewrite. Both arms get one,
    so that stacking is the only difference between them: a stock ``Decoder``
    loops over its layers directly and would ignore the stacks entirely.
    """
    config = registry("debugmodel", attn_backend="sdpa").model
    for layer in config.layers:
        layer.attention.rope = dataclasses.replace(
            layer.attention.rope, max_seq_len=SEQ_LEN
        )
    return config


def _build(
    *,
    stacked: bool,
    param_dtype: str = "float32",
    config=None,
    device: str | torch.device = "cpu",
    world_size: int = 1,
):
    """Build a model the way the trainer does: convert, build, parallelize, init.

    ``stacked`` picks the converter, exactly as ``to_tpu_graph_trainer_config``
    does -- the model arrives from ``build()`` already stacked, and
    ``apply_simple_fsdp`` is given the same arguments either way.

    ``device`` and ``world_size`` exist for ``test_layer_stack_distributed``,
    which builds the same two models on a real mesh.
    """
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world_size,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    training = TrainingConfig(
        mixed_precision_param=param_dtype, mixed_precision_reduce="float32"
    )
    convert = with_tpu_stacked_layers if stacked else _layer_loop
    with torch.device("meta"):
        model = convert(config or _config()).build()
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
    model.to_empty(device=device)
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
        tokens = torch.randint(0, model.config.vocab_size, (2, SEQ_LEN))

        torch.testing.assert_close(
            stacked(tokens), model(tokens), rtol=0, atol=0
        )

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
        tokens = torch.randint(0, plain.config.vocab_size, (2, SEQ_LEN))

        # Perturb the source so a no-op load cannot pass.
        with torch.no_grad():
            for p in plain.parameters():
                p.add_(torch.randn_like(p))
        checkpoint = copy.deepcopy(plain.state_dict())

        stacked.load_state_dict(checkpoint)
        torch.testing.assert_close(stacked(tokens), plain(tokens), rtol=0, atol=0)

    def test_refuses_load_state_dict_with_assign(self):
        """``assign=True`` would load into a tensor the stack never sees.

        It replaces the per-layer view instead of writing through it, so the
        stack keeps its old values and the loaded ones are dropped with the
        views. Silently wrong weights, so it has to raise.
        """
        stacked = _build(stacked=True)
        checkpoint = copy.deepcopy(_build(stacked=False).state_dict())

        with self.assertRaisesRegex(RuntimeError, "assign=True"):
            stacked.load_state_dict(checkpoint, assign=True)


def _toy_layer(dim=4, extra=None, out=None, cache=None, frozen=False):
    layer = nn.Module()
    layer.wq = nn.Linear(dim, out if out is not None else dim, bias=False)
    layer.wq.weight.requires_grad_(not frozen)
    if extra is not None:
        layer.wextra = nn.Linear(dim, extra, bias=False)
    if cache is not None:
        layer.register_buffer("cache", torch.as_tensor(cache))
    return layer


def _toy_layers(*mods):
    return nn.ModuleDict({str(i): m for i, m in enumerate(mods)})


class TestStackable(unittest.TestCase):
    """Every rejection here would otherwise be a silently wrong run.

    A layer that does not match layer 0 still trains: its weights are cast into
    layer 0's stack, or it runs with layer 0's buffers. So each case checks the
    message names the layer and the thing that differs.
    """

    def test_identical_layers_are_stackable(self):
        check_layers_are_stackable(_toy_layers(*[_toy_layer() for _ in range(3)]))
        check_layers_are_stackable(
            _toy_layers(_toy_layer(cache=[1.0]), _toy_layer(cache=[1.0]))
        )

    def test_a_differing_parameter_set_names_the_layer_and_the_parameter(self):
        # Both directions -- extra and missing.
        for label, layers in (
            (
                "extra",
                _toy_layers(_toy_layer(), _toy_layer(extra=8), _toy_layer(extra=16)),
            ),
            ("missing", _toy_layers(_toy_layer(extra=8), _toy_layer(), _toy_layer())),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                self.assertIn("layer 1", str(caught.exception))
                self.assertIn("wextra.weight", str(caught.exception))

    def test_a_same_named_parameter_shows_what_differs(self):
        # Dtype and requires_grad are the quiet ones: the stack is allocated
        # from layer 0, so a layer that disagrees is folded into it in silence.
        odd_dtype = _toy_layer()
        odd_dtype.wq.to(torch.float64)
        for expected, layers in (
            (["(4, 4)", "(6, 4)"], _toy_layers(_toy_layer(), _toy_layer(out=6))),
            (["float64"], _toy_layers(_toy_layer(), odd_dtype)),
            (["False"], _toy_layers(_toy_layer(), _toy_layer(frozen=True))),
        ):
            with self.subTest(expected):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                message = str(caught.exception)
                self.assertIn("wq.weight", message)
                for fragment in expected:
                    self.assertIn(fragment, message)

    def test_a_differing_buffer_is_rejected(self):
        # Same name, shape and dtype: nothing structural to notice, and every
        # layer would run with layer 0's value.
        for label, layers in (
            ("values", _toy_layers(_toy_layer(cache=[1.0]), _toy_layer(cache=[2.0]))),
            ("absent", _toy_layers(_toy_layer(cache=[1.0]), _toy_layer())),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                message = str(caught.exception)
                self.assertIn("layer 1", message)
                self.assertIn("cache", message)

    def test_meta_buffers_are_compared_by_shape_only(self):
        # No values to compare yet; refusing here would reject every model.
        with torch.device("meta"):
            check_layers_are_stackable(
                _toy_layers(_toy_layer(cache=[1.0]), _toy_layer(cache=[2.0]))
            )

    def _moe_model(self, load_balance_coeff: float | None):
        config = qwen3_registry("debugmodel_moe", attn_backend="sdpa").model
        for layer in config.layers:
            layer.moe.load_balance_coeff = load_balance_coeff
        with torch.device("meta"):
            return config.build()

    def test_moe_layers_are_refused(self):
        """An MoE's load-balancing buffers are per-layer, and buffers aren't
        stacked, so every layer would route off layer 0's."""
        with self.assertRaisesRegex(ValueError, "MoE"):
            check_layers_are_stackable(self._moe_model(1e-3).layers)

    def test_moe_layers_without_load_balancing_are_stackable(self):
        """With no coefficient there is no `expert_bias_E`, and nothing reads
        `tokens_per_expert_E`, so sharing layer 0's buffers changes nothing."""
        check_layers_are_stackable(self._moe_model(None).layers)


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestQwen3(unittest.TestCase):
    """The same properties on a second model, to pin that none of this is llama3.

    Stacking reads the layers' own parameters and never their names, so a model
    with a different attention shape, a different norm placement and its own
    RoPE should need no code. That is a claim about the implementation, and the
    only way to keep it true is to check it against something that is not
    llama3.
    """

    def setUp(self):
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

if __name__ == "__main__":
    unittest.main()
