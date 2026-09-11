"""Tests for the recipe configs the READMEs under recipes/ launch by name.

Each recipe README runs a single ``--config=<name>`` and passes no other
training flags, so a renamed or broken config silently breaks a documented
launch command. Rather than restate the recipe list here, these tests discover
it: every ``--module`` / ``--config`` pair inside a recipe README's shell blocks
must name a config that builds, and ``_EXPECTED`` pins the knobs each of those
configs is defined by.
"""

import importlib
import pathlib
import re

from absl.testing import absltest
from absl.testing import parameterized

from torchtitan.experiments.tpu import tpu_flags


_RECIPES_DIR = pathlib.Path(__file__).resolve().parents[1] / "recipes"

_VMEM_64MIB = "--xla_tpu_scoped_vmem_limit_kib=65536"
_VMEM_128MIB = "--xla_tpu_scoped_vmem_limit_kib=131072"
_SPARSECORE_SCHEDULER = (
    "--xla_tpu_enable_sparse_core_collective_aggregator=true"
    " --xla_tpu_enable_latency_hiding_layer_scheduler=true"
    " --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false"
    " --xla_tpu_dvfs_p_state=7"
)
_SPARSECORE_OFFLOAD = (
    "--xla_tpu_enable_sparse_core_collective_aggregator=true"
    " --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true"
    " --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
)

# config name -> (local_batch_size, steps, compile enabled, LIBTPU_INIT_ARGS).
# One row per distinct recipe configuration; a config shared by several slices
# appears once, which is the point of naming it after the hardware.
_EXPECTED = {
    "qwen3_8b_v6e": (
        6, 15, True,
        f"--xla_tpu_use_enhanced_launch_barrier=false {_VMEM_64MIB}",
    ),
    "qwen3_8b_tpu7x": (
        6, 30, True,
        f"{_SPARSECORE_SCHEDULER} --xla_tpu_scoped_vmem_limit_kib=65472",
    ),
    "qwen3_moe_30b_tpu7x_4x4x4": (6, 30, True, _SPARSECORE_SCHEDULER),
    "llama3_8b_v6e_8": (2, 20, True, _VMEM_128MIB),
    "llama3_8b_v6e_256": (4, 20, True, _VMEM_128MIB),
    "llama3_70b_tpu7x": (4, 20, True, _VMEM_128MIB),
    "redacted_3b_v6e_compile": (4, 20, True, _VMEM_64MIB),
    "redacted_3b_v6e_eager": (4, 20, False, _VMEM_64MIB),
    "redacted_3b_v6e_256_eager": (4, 30, False, _VMEM_64MIB),
    "redacted_3b_lora_v6e_8_compile": (1, 20, True, _VMEM_64MIB),
    "redacted_3b_lora_v6e_128_compile": (1, 20, True, _VMEM_64MIB),
    "redacted_3b_lora_v6e_256_compile": (3, 20, True, _VMEM_64MIB),
    "redacted_3b_lora_v6e_eager": (4, 20, False, _VMEM_64MIB),
    "conformer_test_v6e_8_compile": (512, 20, True, ""),
    "conformer_test_v6e_8_eager": (512, 20, False, ""),
    "mamba2_2_7b_256k_tpu7x_2x2x1_lbs8": (8, 100, False, ""),
    "mamba2_2_7b_256k_tpu7x_2x2x1_lbs1": (1, 100, False, ""),
    "flux_dev_v6e_8_float32": (2, 40, True, _VMEM_64MIB),
    "flux_dev_v6e_8_bfloat16": (3, 40, True, _VMEM_64MIB),
    "flux_dev_v6e_256_float32": (2, 40, True, _SPARSECORE_OFFLOAD),
    "flux_dev_v6e_256_bfloat16": (3, 40, True, _SPARSECORE_OFFLOAD),
    "flux_dev_tpu7x_4x4x4_float32": (3, 40, True, _VMEM_64MIB),
    "flux_dev_tpu7x_4x4x4_bfloat16": (5, 40, True, _VMEM_64MIB),
}

# Config sections a launch command must not carry: the recipe belongs in the
# config, so the command line stays at --module / --config.
_TRAINING_FLAG_RE = re.compile(
    r"--(?:training|parallelism|compile|tpu[-_]config|splash[-_]attention[-_]kernel"
    r"|loss[-_]kernel|lora|metrics|dataloader|flux|conformer|redacted|qwen3"
    r"|activation[-_]checkpoint|optimizer|profiler|checkpoint|debug|validator"
    r"|hf[-_]assets[-_]path|libtpu[-_]init[-_]flags|xla[-_]flags)[-_.]"
)
_SHELL_BLOCK_RE = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
_LAUNCH_RE = re.compile(r"--module[= ](\S+?)[\s\\]+.*?--config[= ]([\w.]+)", re.DOTALL)


def _shell_blocks(readme: pathlib.Path) -> list[str]:
  return _SHELL_BLOCK_RE.findall(readme.read_text())


def _discover_launches() -> list[tuple[str, str, str]]:
  """Returns (readme, module, config) for every launch command under recipes/."""
  launches = []
  for readme in sorted(_RECIPES_DIR.rglob("README.md")):
    rel = str(readme.relative_to(_RECIPES_DIR))
    for block in _shell_blocks(readme):
      for module, config in _LAUNCH_RE.findall(block):
        launches.append((rel, module.rstrip('"\\'), config))
  return launches


_LAUNCHES = _discover_launches()


def _build(module_name: str, config_name: str):
  module = importlib.import_module(f"{module_name}.config_registry")
  return getattr(module, config_name)()


class RecipeConfigTest(parameterized.TestCase):

  def test_discovery_found_every_recipe_readme(self) -> None:
    # Guards the parsing itself: a README whose launch command stops matching
    # would otherwise silently drop out of every test below.
    documented = {readme for readme, _, _ in _LAUNCHES}
    expected = {
        str(p.relative_to(_RECIPES_DIR))
        for p in _RECIPES_DIR.rglob("README.md")
    }
    self.assertEqual(documented, expected)

  @parameterized.named_parameters(
      (f"{config}_{i}", readme, module, config)
      for i, (readme, module, config) in enumerate(_LAUNCHES)
  )
  def test_documented_launch_builds_its_config(
      self, readme: str, module: str, config: str
  ) -> None:
    self.assertIn(
        config,
        _EXPECTED,
        f"{readme} launches --config={config}, which _EXPECTED does not pin",
    )
    local_batch_size, steps, compile_enabled, libtpu_init_args = _EXPECTED[config]

    cfg = _build(module, config)

    self.assertEqual(cfg.training.local_batch_size, local_batch_size)
    self.assertEqual(cfg.training.steps, steps)
    self.assertEqual(cfg.compile.enable, compile_enabled)
    self.assertEqual(
        tpu_flags.format_flags(cfg.libtpu_init_flags), libtpu_init_args
    )
    # No recipe needs XLA compiler flags yet; the field exists for ad-hoc
    # debugging (e.g. xla_dump_to), so an accidental default would be a bug.
    self.assertEmpty(cfg.xla_flags)

  def test_every_pinned_config_is_launched_by_a_readme(self) -> None:
    # Catches a recipe config left behind after its README moved on.
    self.assertEqual({config for _, _, config in _LAUNCHES}, set(_EXPECTED))

  @parameterized.named_parameters(
      (str(p.relative_to(_RECIPES_DIR)).replace("/", "_").removesuffix(".md"), p)
      for p in sorted(_RECIPES_DIR.rglob("README.md"))
  )
  def test_launch_commands_pass_no_training_flags(
      self, readme: pathlib.Path
  ) -> None:
    # The whole point of the recipe configs: everything the recipe needs is in
    # the config, so the command line carries --module and --config only.
    for block in _shell_blocks(readme):
      stray = sorted(set(_TRAINING_FLAG_RE.findall(block)))
      self.assertEmpty(
          stray,
          f"{readme.name} passes training flags on the command line: {stray}",
      )

  def test_qwen3_tpu7x_recipe_captures_the_optimizer(self) -> None:
    # The tpu7x README advertises a single XLA graph over backward and the
    # optimizer step. Those knobs come from the qwen3_8b base, so pin them here.
    cfg = _build("torchtitan.experiments.tpu.qwen3", "qwen3_8b_tpu7x")
    self.assertTrue(cfg.tpu_config.use_graph_trainer)
    self.assertEqual(cfg.tpu_config.compile_mode, "aot_fx_trace")
    self.assertTrue(cfg.tpu_config.compile_optimizer)
    self.assertTrue(cfg.tpu_config.optimizer_in_graph)
    self.assertTrue(cfg.parallelism.enable_tpu_topo_mesh)

  def test_qwen3_v6e_recipe_stays_on_the_regular_trainer(self) -> None:
    # The GraphTrainer OOMs on v6e, and SelectiveAC exceeds the XLA activation
    # limits there, so this recipe must keep the regular trainer and FullAC.
    cfg = _build("torchtitan.experiments.tpu.qwen3", "qwen3_8b_v6e")
    self.assertFalse(cfg.tpu_config.use_graph_trainer)
    self.assertEqual(cfg.tpu_config.compile_mode, "layer")
    self.assertEqual(
        type(cfg.activation_checkpoint).__qualname__, "FullAC.Config"
    )


if __name__ == "__main__":
  absltest.main()
