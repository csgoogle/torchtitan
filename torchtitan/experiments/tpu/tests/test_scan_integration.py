# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The adoption API, and the llama3 config that reaches it.

The other scan tests each check one mechanism. This one checks the seam the
user actually touches: `--tpu-config.use-scan` on one side, `scan.over_layers`
on the other. Three things can go wrong there and none of them is loud:

  * order -- FSDP or activation checkpointing walking `model.layers` before the
    stack replaces it, or the in-body gather installed before the mesh exists;
  * silent downgrades -- a combination scan does not support (TP/CP/PP, half an
    FSDP plan) producing a run that is merely wrong instead of an error, or
    `--tpu-config.use-scan` being accepted and then not reaching the spec the
    lane actually runs;
  * the peel knob resolving differently from the kernel config it is supposed
    to track, which shows up as a torch_tpu lowering failure much later.

    ./run_in_container.sh torchtitan.experiments.tpu.tests.test_scan_integration \
        --device=tpu
"""

import contextlib
import inspect
import typing
import unittest.mock as mock

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.config.manager import ConfigManager
from torchtitan.distributed import activation_checkpoint as ac
from torchtitan.experiments.graph_trainer import llama3 as llama3_gt
from torchtitan.experiments.tpu import scan
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.graph_trainer import parallelize_scan
from torchtitan.experiments.tpu.graph_trainer.config import to_tpu_graph_trainer_config
from torchtitan.experiments.tpu.scan import layer_stack
from torchtitan.experiments.tpu.scan import recompute as scan_recompute
from torchtitan.experiments.tpu.scan import reverse as scan_reverse
from torchtitan.experiments.tpu.tests import scan_fixtures
from torchtitan.experiments.tpu.tests.scan_fixtures import DIM, N_LAYERS
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.experiments.tpu import workarounds


class _Dims:
  """A `ParallelDims` stand-in -- `over_layers` reads only these flags."""

  def __init__(self, **enabled):
    for dim in ("tp", "cp", "pp"):
      setattr(self, f"{dim}_enabled", enabled.get(dim, False))


class _Parallelism:

  def __init__(self, use_simple_fsdp=True):
    self.use_simple_fsdp = use_simple_fsdp


@contextlib.contextmanager
def _no_installs():
  """Stubs out the two global patches, so a guard test cannot leak one."""
  with mock.patch.object(scan_reverse, "install"), mock.patch.object(
      scan_recompute, "install"
  ):
    yield


class OverLayersTest(BaseAcceleratorDeviceTest):
  """`scan.over_layers` -- the ordering it enforces and the inputs it rejects."""

  def test_the_stack_exists_inside_the_block_and_the_layers_are_gone(self):
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    original = model.layers

    with _no_installs(), scan.over_layers(model) as stack:
      self.assertIs(model.layers, stack)
      self.assertIsInstance(stack, layer_stack.ScannedLayerStack)
      self.assertEqual(stack.n_layers, N_LAYERS)
      # The stacked parameters are what the caller must hand its FSDP wrap.
      self.assertLen(stack.stacked_parameters(), len(original["0"].state_dict()))

    # No mesh was given, so the exit is a no-op and the stack stays dense.
    for param in stack.stacked_parameters():
      self.assertEqual(param.shape[0], N_LAYERS)

  def test_the_scanned_model_still_computes_the_unrolled_answer(self):
    """The API's own smoke test: adoption must not change the forward."""
    torch.manual_seed(0)
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    torch.manual_seed(0)
    reference = scan_fixtures.ToyModel(N_LAYERS, DIM)
    h = torch.randn(2, 3, DIM, requires_grad=True)

    with _no_installs(), scan.over_layers(model):
      pass

    torch.testing.assert_close(
        model(h), scan_fixtures.unrolled_reference(reference, h)
    )

  def test_the_global_patches_are_installed_as_asked(self):
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    with mock.patch.object(scan_reverse, "install") as reverse_install, (
        mock.patch.object(scan_recompute, "install")
    ) as recompute_install:
      with scan.over_layers(model, recompute_body=True, reverse_backward=False):
        pass

    reverse_install.assert_not_called()
    recompute_install.assert_called_once_with()

  @parameterized.named_parameters(("tp", "tp"), ("cp", "cp"), ("pp", "pp"))
  def test_an_unsupported_parallelism_is_rejected(self, dim):
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    with self.assertRaisesRegex(ValueError, dim.upper()):
      with _no_installs(), scan.over_layers(
          model, parallel_dims=_Dims(**{dim: True})
      ):
        pass

  def test_a_mesh_without_a_mode_is_rejected(self):
    """Half an FSDP plan would give the in-body gather no group to run on."""
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    with self.assertRaisesRegex(ValueError, "dp_mode and mp_policy"):
      with _no_installs(), scan.over_layers(model, dp_mesh=mock.Mock()):
        pass


class PeelFirstLayerTest(BaseAcceleratorDeviceTest):
  """`resolve_peel_first_layer` -- the one config read in the package."""

  def _config(self, **kernel_flags):
    return mock.Mock(splash_attention_kernel=mock.Mock(**kernel_flags))

  @parameterized.named_parameters(
      ("splash", {"use_splash_attention_kernel": True}, True),
      ("tokamax", {"use_tokamax_splash_attention_kernel": True}, True),
      ("neither", {}, False),
  )
  def test_only_a_splash_attention_kernel_needs_the_peel(self, on, expected):
    """A JAX-exported callee is dropped when the body is cloned into the loop."""
    config = self._config(**{
        "use_splash_attention_kernel": False,
        "use_tokamax_splash_attention_kernel": False,
        **on,
    })
    self.assertIs(scan.resolve_peel_first_layer(config), expected)

  @parameterized.named_parameters(("on", "1", True), ("off", "0", False))
  def test_the_env_override_wins_either_way(self, value, expected):
    """So the peel's cost is measurable against an otherwise identical scan."""
    config = self._config(use_splash_attention_kernel=True)
    with mock.patch.dict("os.environ", {"TPU_SCAN_PEEL_FIRST_LAYER": value}):
      self.assertIs(scan.resolve_peel_first_layer(config), expected)


class GraphTrainerWiringTest(BaseAcceleratorDeviceTest):
  """`--tpu-config.use-scan` has to reach the spec the graph_trainer lane runs.

  That spec is a *copy*, taken before `train.py` installs its
  `patched_parallelize` on the original, so the patch carrying `tpu_use_scan`
  never arrives. Before the fix the flag was accepted, logged as enabled, and
  ignored: at 70B the model asked for 166.72 GiB of a 94.75 GiB device while
  every log line said scan was on. The conversion builds the copy, so the
  conversion is where the wiring goes.
  """

  def _parse(self, *extra: str) -> TPUTrainerConfig:
    return typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.llama3",
            "--config=llama3_debugmodel",
            "--parallelism.use_simple_fsdp",
            *extra,
        ]),
    )

  def _convert(self, *extra: str):
    return to_tpu_graph_trainer_config(
        self._parse(*extra),
        llama3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

  def test_use_scan_installs_the_scan_parallelize_fn(self):
    scanned = self._convert("--tpu_config.use_scan")
    # Compared against the un-scanned conversion rather than against a name:
    # the failure being guarded is "the graph_trainer default survived", and
    # only the two side by side can show that.
    baseline = self._convert()

    self.assertIsNot(
        scanned.model_spec.parallelize_fn, baseline.model_spec.parallelize_fn
    )
    self.assertIn(
        "parallelize_llama_scan",
        inspect.getsource(scanned.model_spec.parallelize_fn),
    )
    self.assertNotIn(
        "parallelize_llama_scan",
        inspect.getsource(baseline.model_spec.parallelize_fn),
    )

  def test_use_scan_on_another_model_is_rejected(self):
    """Only llama3 has a scan-aware parallelize_fn; the rest would run unrolled."""
    config = self._parse("--tpu_config.use_scan")
    config.model_spec.name = "qwen3"
    with self.assertRaisesRegex(ValueError, "llama3 only"):
      to_tpu_graph_trainer_config(
          config, llama3_gt.model_registry, target_cls=TpuGraphTrainer.Config
      )

  @parameterized.named_parameters(("on", "1"), ("off", "0"))
  def test_the_peel_is_resolved_at_conversion_time(self, peel):
    """Bound into the closure, not read again when parallelize runs.

    A value read late would attribute the peel's cost to the wrong arm of the
    A/B the env override exists for.
    """
    with mock.patch.dict("os.environ", {"TPU_SCAN_PEEL_FIRST_LAYER": peel}):
      graph_config = self._convert("--tpu_config.use_scan")
      bound = inspect.getclosurevars(
          graph_config.model_spec.parallelize_fn
      ).nonlocals

    self.assertEqual(bound["peel_first_layer"], peel == "1")


class ParallelizeScanTest(BaseAcceleratorDeviceTest):
  """`parallelize_llama_scan` -- the four steps, in the order that matters."""

  def _run(self, memory_policy="default"):
    """Runs the lane's parallelize with everything but the scan block mocked.

    Returns the `apply_simple_fsdp` mock -- which is where the ordering shows
    up, being called inside the `over_layers` block on the stacked model -- and
    the stack itself.
    """
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    with contextlib.ExitStack() as stack:
      patch = lambda name, **kw: stack.enter_context(  # pylint: disable=g-long-lambda
          mock.patch.object(parallelize_scan, name, **kw)
      )
      patch("resolve_dp_plan", return_value=(None, None, None))
      patch("annotate_llama")
      patch("apply_compile", side_effect=lambda m, **_: m)
      fsdp = patch("apply_simple_fsdp", side_effect=lambda m, **_: m)
      stack.enter_context(mock.patch.object(scan_recompute, "install"))
      stack.enter_context(mock.patch.object(scan_reverse, "install"))

      parallelize_scan.parallelize_llama_scan(
          model,
          parallel_dims=_Dims(),
          training=mock.Mock(),
          parallelism=_Parallelism(),
          compile_config=mock.Mock(memory_policy=memory_policy),
          ac_config=ac.SelectiveAC.Config(),
          dump_folder="/tmp",
      )
    return fsdp, model.layers

  def test_fsdp_skips_the_stacked_parameters(self):
    """Otherwise they are sharded twice: once here, once in the scan body."""
    fsdp, _ = self._run()

    ignored = fsdp.call_args.kwargs["ignored_params"]
    self.assertLen(ignored, len(scan_fixtures.ToyBlock(DIM).state_dict()))
    for param in ignored:
      self.assertEqual(param.shape[0], N_LAYERS)

  @parameterized.named_parameters(("full", "full", True), ("default", "default", False))
  def test_memory_policy_full_recomputes_the_body(self, policy, expected):
    """This lane spells activation checkpointing `memory_policy`, not `ac_config`."""
    _, stack = self._run(memory_policy=policy)
    self.assertIs(stack.recompute_body, expected)


class ModulePatchReachTest(BaseAcceleratorDeviceTest):
  """A model-wide module patch has to reach the block the scan body runs.

  The stack keeps its per-layer blocks unregistered so their `meta` parameters
  stay out of `state_dict()`, which also hides them from `model.modules()`.
  `workarounds.apply_patches` runs after `parallelize_fn`, so the splash
  attention patch walked a model whose only remaining module types were
  `SimpleFSDPScannedLayerStack` and friends and raised "NO matching attention
  modules" -- 8B llama3, where splash is on by default.
  """

  def test_the_walk_descends_into_the_scan_template(self):
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    stack = layer_stack.stack_layers(model)

    found = dict(workarounds._all_modules(model))  # pylint: disable=protected-access

    self.assertNotIn(
        stack.scan_template,
        [m for _, m in model.named_modules()],
        "precondition: the template must be invisible to a plain walk",
    )
    self.assertIn(stack.scan_template, found.values())
    self.assertIn(stack.scan_template.wi, found.values())

  def test_patching_the_template_changes_what_the_body_runs(self):
    """The patch is only useful if the scan actually calls the patched module."""
    model = scan_fixtures.ToyModel(N_LAYERS, DIM)
    stack = layer_stack.stack_layers(model)
    for _, module in workarounds._all_modules(model):  # pylint: disable=protected-access
      if isinstance(module, scan_fixtures.ToyBlock):
        module.forward = lambda h, *_, **__: h * 0 + 7.0

    out = stack(torch.ones(2, DIM, requires_grad=True))

    torch.testing.assert_close(out, torch.full((2, DIM), 7.0))


if __name__ == "__main__":
  absltest.main()
