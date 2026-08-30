import time
import jax
import optax
from jax.tree_util import tree_map
from jax.sharding import NamedSharding
from torch_xla2 import interop
import torchax
import torchtitan.tools.logging

P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


def make_split_train_step(model_fn, loss_fn, optax_optimizer, remat_policy=None):
  """Analog of ``torchax.train.make_train_step`` that returns the forward+
  backward (`grad_fn`) and the optimizer update (`opt_fn`) as *separate*
  callables, intended to be ``jax.jit``-compiled independently and chained in
  Python.

  The default ``make_train_step`` returns one monolithic ``step()`` which
  ``compile_step_func`` turns into a single HLO module covering forward,
  backward, and optimizer. Splitting here produces two smaller HLO modules
  — directly comparable to how ``torchtpu`` with ``torch.compile(model,
  components=["model"])`` compiles the model forward/backward and runs the
  optimizer outside compile.
  """
  env = torchax.default_env()

  def loss(weights, buffers, args, label):
    with env, jax.named_scope("compute_loss"):
      res = model_fn(weights, buffers, args)
      return loss_fn(res, label)

  grad_fn = interop.jax_value_and_grad(loss)

  def grad_step(weights, buffers, args, label):
    with jax.named_scope("compute_gradient"):
      return grad_fn(weights, buffers, args, label)

  def opt_step(weights, gradient, opt_state):
    with jax.named_scope("optimizer_updates"):
      updates, opt_state = interop.call_jax(  # pyrefly: ignore[not-iterable]
          optax_optimizer.update, gradient, opt_state, weights
      )
      weights = interop.call_jax(optax.apply_updates, weights, updates)
    return weights, opt_state

  return grad_step, opt_step


def compile_split_step_func(
    grad_step, opt_step, weights, buffers, opt_state, inputs, labels, mesh
):
  """Compile ``grad_step`` and ``opt_step`` as two separate jitted functions
  and return a Python composition. The outer composition is not jitted — each
  piece runs as an independent XLA program, so HLO is split cleanly along
  fwd+bwd / optimizer boundary (mirroring torchtpu's compile granularity)."""
  grad_step, opt_step, weights, buffers, opt_state, inputs, labels = (
      interop.jax_view(
          (grad_step, opt_step, weights, buffers, opt_state, inputs, labels)
      )
  )
  wshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, weights)
  oshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, opt_state)
  loss_sharding = NamedSharding(mesh, P())

  logger.info("Compiling grad step (fwd+bwd) for first iteration.")
  start = time.perf_counter()
  grad_lowered = jax.jit(
      grad_step,
      donate_argnums=(),
      out_shardings=(loss_sharding, wshardings),
  ).lower(weights, buffers, inputs, labels)
  grad_text = grad_lowered.as_text()
  logger.info("grad_step HLO size: %.3f MB", len(grad_text) / 1e6)
  grad_compiled = grad_lowered.compile()
  logger.info("grad_step compile took %.2fs", time.perf_counter() - start)

  logger.info("Compiling opt step (optimizer update) for first iteration.")
  start = time.perf_counter()
  # For opt_step compile we need a sample gradient; reuse weights' structure.
  opt_lowered = jax.jit(
      opt_step,
      donate_argnums=(0, 2),
      out_shardings=(wshardings, oshardings),
  ).lower(weights, weights, opt_state)
  opt_text = opt_lowered.as_text()
  logger.info("opt_step HLO size: %.3f MB", len(opt_text) / 1e6)
  opt_compiled = opt_lowered.compile()
  logger.info("opt_step compile took %.2fs", time.perf_counter() - start)

  grad_compiled_tv = interop.torch_view(grad_compiled)
  opt_compiled_tv = interop.torch_view(opt_compiled)

  def split_step(weights, buffers, opt_state, inputs, labels):
    loss_val, gradient = grad_compiled_tv(weights, buffers, inputs, labels)
    weights, opt_state = opt_compiled_tv(weights, gradient, opt_state)
    return loss_val, weights, opt_state

  return split_step


def compile_step_func(step, weights, buffers, opt_state, inputs, labels, mesh):
  """Compiles a single training step function using JAX jit.

  Args:
    step: The step function to be compiled.
    weights: The model weights.
    buffers: The model buffers.
    opt_state: The optimizer state.
    inputs: Training data input.
    labels: Training data label.
    mesh: The JAX device mesh for sharding.

  Returns:
    A compiled and pytorch-view step function.
  """
  step, weights, buffers, opt_state, inputs, labels = interop.jax_view(
      (step, weights, buffers, opt_state, inputs, labels))
  wshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, weights)
  bshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, buffers)
  oshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, opt_state)
  logger.info('Compiling train step for first iteration.')
  start = time.perf_counter()
  lowered = jax.jit(
      step,
      donate_argnums=(0, 2),
      #in_shardings=shardings,
      out_shardings=(NamedSharding(mesh, P()), wshardings, oshardings),
  ).lower(weights, buffers, opt_state, inputs, labels)

  logger.debug('program size: %.4f m chars', len(lowered.as_text()) / 1e6)
  step_compiled = lowered.compile()
  end = time.perf_counter()
  logger.info('Compilation done. It took %.4f seconds', end - start)
  for co in step_compiled.cost_analysis():  # pyrefly: ignore[not-iterable]
    logger.debug('Cost analysis: %s', co)
  return interop.torch_view(step_compiled)
