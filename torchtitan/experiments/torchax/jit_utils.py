import time
import jax
from jax.tree_util import tree_map
from jax.sharding import NamedSharding
from torch_xla2 import interop
import torchtitan.tools.logging

P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


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
  for co in step_compiled.cost_analysis():
    logger.debug('Cost analysis: %s', co)
  return interop.torch_view(step_compiled)
