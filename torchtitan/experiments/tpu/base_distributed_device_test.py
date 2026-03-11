"""Base class and helper functions for distributed accelerator tests."""

import enum
import os
import tempfile
from typing import Any, Callable, Dict, Tuple, Type

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.google import distributed as gdist
import torchtitan.config
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import distributed_utils
from torchtitan.experiments.tpu import numerical_validation
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import train_minimal
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
from torchtitan.tools import utils


# Make Fairscale optional since we are not using it in all tests.
try:
  from fairscale.nn import model_parallel
  from fairscale.nn.model_parallel import layers as fairscale_layers
  HAS_FAIRSCALE = True
except ImportError:
  HAS_FAIRSCALE = False

ParallelismFunc = Callable[[nn.Module], nn.Module | None]


class InputDistribution(enum.Enum):
  """Input distribution for distributed tests."""
  REPLICATE = "replicate"   # All ranks see all data (e.g. TP)
  SPLIT_BATCH = "split_batch"  # Rank i only sees batch[i::world_size] (e.g. FSDP)


# Gathering helper functions (shared by unit test runner and training state
# validator)
def _gather_fairscale_weights(
    model: nn.Module, tensor: torch.Tensor, name: str, world_size: int
) -> torch.Tensor:
  """Helper to manually all-gather a Fairscale param/grad."""
  if not HAS_FAIRSCALE:
    raise ImportError("use_fairscale=True but fairscale is not installed.")

  # Gather all shards
  # We assume the default model parallel group setup by Fairscale
  try:
    group = model_parallel.initialize.get_model_parallel_group()
  except (ImportError, AttributeError):
    # Fallback if initialization hasn't happened
    group = dist.group.WORLD

  gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
  dist.all_gather(gathered, tensor, group=group)

  # Identify module type
  module_name = name.rsplit(".", 1)[0]

  # Use get_submodule for safer lookup
  try:
    parent = model.get_submodule(module_name)
  except AttributeError:
    modules = dict(model.named_modules())
    parent = modules.get(module_name, None)

  if parent is None:
    # Assume replicated as fallback
    return gathered[0]

  # Stitch based on type
  if isinstance(parent, fairscale_layers.ColumnParallelLinear):
    return torch.cat(gathered, dim=0)  # Shards Output (Dim 0)

  elif isinstance(parent, fairscale_layers.RowParallelLinear):
    if "weight" in name:
      return torch.cat(gathered, dim=1)  # Shards Input (Dim 1)
    return gathered[0]  # Bias is replicated

  elif isinstance(parent, fairscale_layers.ParallelEmbedding):
    return torch.cat(gathered, dim=1)  # Shards Dim 1 (Hidden)

  # Default/Replicated
  return gathered[0]


def _gather_distributed_tensor(
    model: nn.Module,
    tensor: torch.Tensor | DTensor,
    name: str,
    world_size: int,
    use_fairscale: bool = False,
) -> torch.Tensor:
  """Unshards a tensor (DTensor or Fairscale) to CPU."""

  # Handle Fairscale (manual stitching)
  if use_fairscale:
    return (
        _gather_fairscale_weights(model, tensor, name, world_size)
        .detach()
        .cpu()
    )

  # Handle DTensor (automatic stitching)
  if isinstance(tensor, DTensor):
    return tensor.full_tensor().detach().cpu()

  # Should be one of the above, otherwise its a standard tensor and
  # replicated shard so we can just take local copy.
  return tensor.detach().cpu()


class DistributedUnitTestRunner:
  """Runs distributed parity tests between a Single-Device CPU model and a Parallel Device model.

  Works with DTensor- or Fairscale-based parallelism set ups

  This class manages:
  - Creating single-device CPU model as a reference.
  - Creating parallelized device model to test against reference.
  - Ensuring test/ref models start with identical weights.
  - Applying arbitrary parallelization transforms.
  - Running synchronized Forward/Backward/Step checks.
  """

  def __init__(
      self,
      device: torch.device,
      rank: int,
      world_size: int,
      model_class: Type[nn.Module],
      model_args: Any,
      parallelism_func: ParallelismFunc,
      input_distribution: InputDistribution = InputDistribution.REPLICATE,
      use_meta_init: bool = True,
      use_fairscale: bool = False,
  ):
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.model_class = model_class
    self.model_args = model_args
    self.input_distribution = input_distribution
    self.use_meta_init = use_meta_init
    self.use_fairscale = use_fairscale

    # Apply parallelism to the model.
    self.parallel_model = None
    self._create_parallel_model(parallelism_func)

    # Create CPU reference model.
    self.reference_model = None
    self._create_reference_from_parallel()

    # Track current step for parity checks.
    self.current_step = 0

  def run_forward_parity(
      self,
      batch_size: int,
      seq_len: int,
      atol: float = 1e-3,
      rtol: float = 1e-3
  ):
    """Verifies that reference and parallel model produce equivalent outputs on forward pass."""
    if self.reference_model is None:
      raise ValueError("Reference model not initialized. To run test, call "
                       "`apply_parallelism`to parallelize the device model "
                       "and create a reference on CPU.")

    global_input, global_target = self._generate_global_batch(
        batch_size, seq_len
    )
    local_input, _ = self._get_local_batch_view(global_input, global_target)

    reference_out = None

    # Run reference model (Rank 0 only)
    if self.rank == 0:
      with torch.no_grad():
        reference_out, _ = self._run_step(
            self.reference_model, global_input, backward=False
        )

    with torch.no_grad():
      parallel_out, _ = self._run_step(
          self.parallel_model, local_input.to(self.device), backward=False
      )

    # Helper to move tensor/dtensor to CPU
    parallel_out_cpu = self._gather_output_to_cpu(parallel_out)

    if self.rank == 0:
      test_utils.check_equivalence(
          tensor_test=parallel_out_cpu,
          tensor_reference=reference_out,
          check_name=f"Forward Pass (Rank {self.rank})",
          step=self.current_step,
          atol=atol,
          rtol=rtol,
          test_label="Parallel",
          ref_label="Single Device",
      )
      logging.info("[Rank %s] Forward parity passed.", self.rank)

  def run_backward_parity(
      self,
      batch_size: int,
      seq_len: int,
      atol_loss: float = 1e-3,
      rtol_loss: float = 1e-3,
      atol_grad: float = 1e-3,
      rtol_grad: float = 1e-3,
  ):
    """Verifies the reference and parallel models produce identical loss and gradients."""
    if self.reference_model is None:
      raise ValueError(
          "Reference model not initialized. To run test, call "
          "`apply_parallelism`to parallelize the device model "
          "and create a reference on CPU."
      )
    inputs_global, targets_global = self._generate_global_batch(
        batch_size, seq_len
    )
    inputs_local, targets_local = self._get_local_batch_view(
        inputs_global, targets_global
    )
    inputs_device = inputs_local.to(self.device)
    targets_device = targets_local.to(self.device)

    reference_loss = None
    # Run reference model (Rank 0 only)
    if self.rank == 0:
      _, reference_loss = self._run_step(
          self.reference_model, inputs_global, targets_global, backward=True
      )

    # Run parallel model (all ranks)
    _, parallel_loss = self._run_step(
        self.parallel_model, inputs_device, targets_device, backward=True
    )

    # If we split the batch, local loss is just a slice. We must average across
    # ranks to match the global reference loss.
    if self.input_distribution == InputDistribution.SPLIT_BATCH:
      loss_to_check = parallel_loss.detach().clone()
      dist.all_reduce(loss_to_check, op=dist.ReduceOp.AVG)
    else:
      loss_to_check = parallel_loss

    # Compare Loss
    if self.rank == 0:
      test_utils.check_equivalence(
          tensor_test=loss_to_check.cpu(),
          tensor_reference=reference_loss,
          check_name=f"Backward Loss (Rank {self.rank})",
          step=self.current_step,
          atol=atol_loss, rtol=rtol_loss,
          test_label="Parallel",
          ref_label="Single Device",
      )

    self._check_gradients(atol_grad, rtol_grad)

    logging.info("[Rank %s] Backward parity passed.", self.rank)
    self.current_step += 1

  def _create_parallel_model(self, func: Callable[[nn.Module], None]):
    """Creates a parallel model and applies the parallelism function."""
    if self.use_meta_init:
      with torch.device("meta"):
        self.parallel_model = self.model_class(self.model_args)
    else:
      self.parallel_model = self.model_class(self.model_args).to(self.device)

    # Capture return value if parallelism_func does not make change in place.
    modified_model = func(self.parallel_model)
    if modified_model is not None:
      self.parallel_model = modified_model

    # Materialize weights if they are still on Meta device
    if any(p.device.type == "meta" for p in self.parallel_model.parameters()):
      self.parallel_model.to_empty(device=self.device)
      with torch.device(self.device):
        self.parallel_model.init_weights()

  def _create_reference_from_parallel(self):
    """Creates a CPU reference model and loads weights from the parallel model."""
    self.reference_model = self.model_class(self.model_args).to("cpu")

    parallel_state = self.parallel_model.state_dict()
    cpu_state_dict = {}

    with torch.no_grad():
      for key, param in parallel_state.items():
        cpu_state_dict[key] = self._gather_weights_to_cpu_wrapper(param, key)

    self.reference_model.load_state_dict(cpu_state_dict)
    # Ensure gradients are enabled on reference model params.
    for p in self.reference_model.parameters():
      p.requires_grad_(True)

  def _generate_global_batch(
      self, batch_size: int, seq_len: int
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates the full global batch on CPU (deterministically)."""
    torch.manual_seed(1 + self.current_step)

    global_input = torch.randint(
        0, self.model_args.vocab_size, (batch_size, seq_len), device="cpu"
    )
    global_target = torch.randn(
        batch_size, seq_len, self.model_args.vocab_size, device="cpu"
    )
    return global_input, global_target

  def _get_local_batch_view(
      self, global_input: torch.Tensor, global_target: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slices or replicates the global batch for the current rank."""

    if self.input_distribution == InputDistribution.REPLICATE:
      # TP: Everyone sees the full global batch
      return global_input, global_target

    elif self.input_distribution == InputDistribution.SPLIT_BATCH:
      # FSDP: Slice the global batch for this rank
      batch_size = global_input.size(0)
      assert batch_size % self.world_size == 0

      local_bs = batch_size // self.world_size
      start = self.rank * local_bs
      end = start + local_bs

      return global_input[start:end], global_target[start:end]

    raise ValueError(f"Unknown distribution: {self.input_distribution}")

  def _run_step(
      self,
      model: nn.Module,
      inputs: torch.Tensor,
      targets: torch.Tensor = None,
      backward: bool = False
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs a forward pass and optionally a backward pass."""
    if backward:
      model.zero_grad()

    output = model(inputs)
    loss = None

    if backward and targets is not None:
      loss = nn.MSELoss()(output, targets)
      loss.backward()

    return output, loss

  def _check_gradients(self, atol, rtol):
    """Standard DTensor gradient check."""
    ref_map = {}
    if self.rank == 0:
      ref_map = dict(self.reference_model.named_parameters())

    for name, par_param in self.parallel_model.named_parameters():
      if par_param.grad is None:
        continue

      full_grad = self._gather_weights_to_cpu_wrapper(par_param.grad, name)

      if self.rank == 0 and name in ref_map and ref_map[name].grad is not None:
        test_utils.check_equivalence(
            tensor_test=full_grad,
            tensor_reference=ref_map[name].grad,
            atol=atol,
            rtol=rtol,
            check_name=f"Gradient {name}",
            step=self.current_step,
        )

  def _gather_output_to_cpu(
      self,
      output: torch.Tensor | DTensor,
  ) -> torch.Tensor:
    """Standardizes gathering outputs from DTensor or Tensor to CPU."""
    if isinstance(output, DTensor):
      # DTensor knows how to unshard itself
      return output.full_tensor().cpu()

    # If it's local tensor, behavior depends on the test
    # If TP output is sharded on sequence dim (SP) but is not DTensor,
    # we might need manual gathering logic here.
    # Not needed currently since Fairscale implements pure TP.
    return output.cpu()

  def _gather_weights_to_cpu_wrapper(
      self,
      tensor: torch.Tensor | DTensor,
      name: str
  ) -> torch.Tensor:
    """Wraps module-level gather function with instance config."""
    # Fairscale logic: Only gather manually if we are in TP (Replicate) mode.
    # If FSDP (split batch), Fairscale state_dict handles it automatically,
    # so we pass use_fairscale=False to the helper to skip manual gathering.
    pass_fairscale = self.use_fairscale and (
        self.input_distribution == InputDistribution.REPLICATE
    )

    return _gather_distributed_tensor(
        self.parallel_model,
        tensor,
        name,
        self.world_size,
        use_fairscale=pass_fairscale,
    )


def _capture_distributed_state(
    model: nn.Module,
    loss: torch.Tensor,
    world_size: int,
    use_fairscale: bool,
    input_distribution: InputDistribution = InputDistribution.REPLICATE,
) -> Dict[str, Any]:
  """Captures a static snapshot of the distributed model's parameters, gradients, and loss."""

  # Unwrap DTensor if present (Keep on device)
  if isinstance(loss, DTensor):
    loss_local = loss.full_tensor()
  else:
    loss_local = loss

  # Detach and clone (on device)
  loss_on_device = loss_local.detach().clone()

  # Perform reduction if needed (on device)
  if input_distribution == InputDistribution.SPLIT_BATCH and world_size > 1:
    # Average across ranks (e.g. for FSDP split batch)
    dist.all_reduce(loss_on_device, op=dist.ReduceOp.AVG)

  # Move result
  loss_cpu = loss_on_device.cpu()

  state = {
      "loss": loss_cpu,
      "params": {},
      "grads": {},
  }

  for name, param in model.named_parameters():
    state["params"][name] = _gather_distributed_tensor(
        model, param, name, world_size, use_fairscale
    )

    if param.grad is not None:
      state["grads"][name] = _gather_distributed_tensor(
          model, param.grad, name, world_size, use_fairscale
      )

  return state


def config_to_input_distribution(
    config: torchtitan.config.JobConfig,
) -> InputDistribution:
  """Determines the input distribution based on the config.

  Args:
    config: The job configuration.

  Returns:
    The determined InputDistribution based on the data parallel shard degree.
  """
  dp_degree = config.parallelism.data_parallel_shard_degree
  if dp_degree > 1:
    return InputDistribution.SPLIT_BATCH
  else:
    return InputDistribution.REPLICATE


# Chose for wrapper to be class so that output_path can written to/read from
# outside the function.
# (alternative is to use functools.partial)
class DistributedTrainWithRecorder:
  """Runs a distributed training job while recording model state history to a temporary file."""

  def __init__(self, output_path: str):
    self.output_path = output_path

  def __call__(self, device, rank, world_size, config):

    use_fairscale = (
        hasattr(config, "tpu_config") and config.tpu_config.use_fairscale
    )
    dist_type = config_to_input_distribution(config)

    # Define how to capture state in current distributed context
    def dist_capture_fn(model, loss):
      return _capture_distributed_state(
          model, loss, world_size, use_fairscale, dist_type
      )

    # Setup recorder callback
    recorder = numerical_validation.StateRecorderCallback(capture_fn=dist_capture_fn)

    trainer = train_minimal.TrainerMinimal(
        device=device,
        rank=rank,
        world_size=world_size,
        job_config=config,
        step_callback=recorder.on_step,
    )

    # Init State Capture (Manual call to prevent deadlock)
    init_loss = torch.tensor(0.0, device=device)
    init_state = dist_capture_fn(trainer.model, init_loss)

    if rank == 0:
      recorder.history["init"] = init_state

    trainer.train()

    if rank == 0:
      torch.save(recorder.get_history(), self.output_path)


class BaseDistributedDeviceTest(parameterized.TestCase):
  """Base class for tests that need to run on different distributed devices.

  This class sets up `self.num_devices` based on the `--num_devices` flag and
  `self.accelerator_device_type` based on the `--device` flag.
  """

  def setUp(self):
    super().setUp()
    torch.compiler.reset()
    self.num_devices = test_utils._NUM_DEVICES.value
    self.accelerator_device_type = test_utils._DEVICE.value

    # Ensure Hugging Face datasets use a directory with enough storage
    # (e.g. /tmp). The default cache in `~/.cache` is too small in some
    # environments.
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        tempfile.gettempdir(), "huggingface_cache"
    )

    # Set torch and torch.cuda seeds to ensure deterministic behavior.
    # Note that tests would set it by default to 301, unless overridden:
    # [internal link]
    if absltest.FLAGS["test_random_seed"].present:
      seed = absltest.FLAGS.test_random_seed
    else:
      seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if not using GPU
    print(f"Torch initial seed: {torch.initial_seed()}")

  def test_gpu_available(self):
    if self.accelerator_device_type == "cuda":
      if not torch.cuda.is_available():
        self.fail("GPU requested but CUDA not available.")

  def _test_train_distributed(
      self,
      config_args: list[str],
      use_fairscale: bool,
      data_parallel_shard_degree: int | None = None,
      tensor_parallel_degree: int | None = None,
      skip_devices: list[device_type.AcceleratorDeviceType] | None = None,
      start_trainer: Callable[
          [
              torchtitan.config.JobConfig | tpu_job_config_module.TPUJobConfig,
          ],
          Any,
      ] = None,
      run_init_process_group: bool = True,
      enable_compile: bool = False,
  ):
    """Tests the minimal distributed training setup for torchtitan models."""
    if skip_devices and self.accelerator_device_type in skip_devices:
      self.skipTest(
          f"Skipping test for device type: {self.accelerator_device_type}"
      )
      return

    if enable_compile:
      if self.accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
        self.skipTest("Skipping CPU test for torch.compile: b/327271919")
        return
      config_args.append("--compile.enable")
      if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
        config_args.append("--compile.backend=tpu")

    if tensor_parallel_degree is not None:
      if tensor_parallel_degree == -1:
        tensor_parallel_degree = self.num_devices
      config_args.append(
          f"--parallelism.tensor_parallel_degree={tensor_parallel_degree}"
      )
    if data_parallel_shard_degree is not None:
      if data_parallel_shard_degree == -1:
        data_parallel_shard_degree = self.num_devices
      config_args.append(
          f"--parallelism.data_parallel_shard_degree={data_parallel_shard_degree}"
      )

    if use_fairscale:
      config_args.append("--tpu_config.use_fairscale")

    config_manager = torchtitan.config.ConfigManager(
        tpu_job_config_module.TPUJobConfig
    )

    config = config_manager.parse_args(config_args)
    config.job.dump_folder = tempfile.gettempdir()

    utils.set_device_type(self.accelerator_device_type.value)
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      # Construct the environment before spawning workers.
      distributed_utils.maybe_init_distributed(self.num_devices)

    gdist.torchrun(start_trainer, nproc_per_node=self.num_devices)(config)

  def _run_trainer_distributed_parity_test(
      self,
      config_args: list[str],
      use_fairscale: bool,
      tensor_parallel_degree: int | None = None,
      data_parallel_shard_degree: int | None = None,
      skip_devices: list[device_type.AcceleratorDeviceType] | None = None,
      loss_atol: float = 1e-3,
      loss_rtol: float = 1e-3,
      grad_atol: float = 5e-3,  # grads can be noisier in distributed setting
      grad_rtol: float = 5e-3,
      param_atol: float = 5e-3,
      param_rtol: float = 5e-3,
      enable_compile: bool = False,
  ):
    """Orchestrates a distributed parity test: Record Distributed -> Verify on CPU."""
    # Setup temporary file for data transfer.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      temp_path = tmp_file.name

    try:
      logging.info("[Phase 1] Running Distributed Training on device (Recording)...")

      # Use wrapper class to run the trainer and record the state to the file.
      self._test_train_distributed(
          config_args=config_args,
          use_fairscale=use_fairscale,
          data_parallel_shard_degree=data_parallel_shard_degree,
          tensor_parallel_degree=tensor_parallel_degree,
          skip_devices=skip_devices,
          start_trainer=DistributedTrainWithRecorder(
              output_path=temp_path
          ),
          enable_compile=enable_compile,
      )

      # Load recorded data from the file (if non-empty; otherwise fail)
      if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
        recorded_history = torch.load(temp_path)
        logging.info(
            "Retrieved history for %s steps (including init).",
            len(recorded_history),
        )
      else:
        self.fail(
            "Distributed run finished but returned no data (File"
            " empty/missing)."
        )

    finally:
      # Cleanup temporary file
      if os.path.exists(temp_path):
        os.remove(temp_path)

    # Setup CPU Verification Run
    logging.info("[Phase 2] Running CPU Training (Verifying)...")

    # Extract initialization state (must exist)
    if "init" not in recorded_history:
      self.fail("Recorded history missing 'init' state. Cannot sync CPU model.")
    init_state = recorded_history.pop("init")

    # Prepare CPU Configuration
    # We parse the original args but force parallelism to 1 and device to CPU
    config_manager = torchtitan.config.ConfigManager(
        tpu_job_config_module.TPUJobConfig
    )
    cpu_config = config_manager.parse_args(config_args)

    # Force overrides for single-device CPU execution
    cpu_config.parallelism.tensor_parallel_degree = 1
    cpu_config.parallelism.data_parallel_shard_degree = 1
    cpu_config.compile.enable = False

    # Setup validator callback with the recorded history.
    validator = numerical_validation.StateValidatorCallback(
        recorded_history=recorded_history,
        capture_fn=numerical_validation.capture_local_state,
        loss_atol=loss_atol, loss_rtol=loss_rtol,
        grad_atol=grad_atol, grad_rtol=grad_rtol,
        param_atol=param_atol, param_rtol=param_rtol,
    )

    with validator:
      # Initialize CPU Trainer
      trainer_cpu = train_minimal.TrainerMinimal(
          device=torch.device("cpu"),
          rank=0,
          world_size=1,
          job_config=cpu_config,
          step_callback=validator.on_step,
      )

      # Load the captured distributed weights into the CPU model
      current_cpu_dict = trainer_cpu.model.state_dict()

      # We only load params that exist in the CPU model
      # (Distributed run might have extra buffers, but params must match)
      for name, param in init_state["params"].items():
        if name in current_cpu_dict:
          with torch.no_grad():
            current_cpu_dict[name].copy_(param)
        else:
          logging.warning(
              "Param %s from distributed run not found in CPU model.", name
          )

      logging.info("CPU model initialized with distributed weights.")

      # Run CPU training with validation callback to compare against reference.
      trainer_cpu.train()

    logging.info("Parity Test Passed Successfully.")
