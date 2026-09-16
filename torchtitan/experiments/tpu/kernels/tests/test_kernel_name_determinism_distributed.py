"""Tests that custom kernel op names are deterministic across subprocesses."""

import traceback
from typing import Any, Callable

from absl import logging
from absl.testing import absltest
import torch.multiprocessing as mp
from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss
from torchtitan.experiments.tpu.kernels import splash_attention

_TIMEOUT_SECONDS = 120


def _run_worker(
    q: mp.Queue,
    fn: Callable[..., tuple[str, str]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
  """Worker process entrypoint to invoke op name function and put result in queue."""
  try:
    fwd, bwd = fn(*args, **kwargs)
    q.put((True, (fwd, bwd)))
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error(
        "Exception in _run_worker for %s: %s\n%s",
        fn,
        e,
        traceback.format_exc(),
    )
    q.put((False, str(e)))


class KernelNameDeterminismTest(absltest.TestCase):
  """Tests determinism of generated kernel op names across processes."""

  def _get_kernel_names_in_subprocess(
      self,
      fn: Callable[..., tuple[str, str]],
      *args: Any,
      **kwargs: Any,
  ) -> tuple[str, str]:
    """Runs the provided function in an isolated spawned subprocess and returns op names."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_worker, args=(q, fn, args, kwargs))
    p.start()
    success, res = q.get(timeout=_TIMEOUT_SECONDS)
    p.join()
    self.assertTrue(success, f"Subprocess execution failed: {res}")
    return res

  def test_linear_softmax_cross_entropy_loss_kernel_names_are_deterministic_across_subprocesses(
      self,
  ):
    """Tests that linear loss kernel names match across spawned subprocesses."""
    # pylint: disable=protected-access
    fn = linear_softmax_cross_entropy_loss._get_loss_op_names
    # pylint: enable=protected-access
    args = ("sum", "mosaic_tpu", 128, 128, 128)

    fwd1, bwd1 = self._get_kernel_names_in_subprocess(fn, *args)
    fwd2, bwd2 = self._get_kernel_names_in_subprocess(fn, *args)

    self.assertEqual(fwd1, fwd2)
    self.assertEqual(bwd1, bwd2)

  def test_splash_attention_kernel_names_are_deterministic_across_subprocesses(
      self,
  ):
    """Tests that splash attention kernel names match across spawned subprocesses."""
    # pylint: disable=protected-access
    fn = splash_attention._get_splash_op_names
    # pylint: enable=protected-access
    splash_args = dict(
        seq_len=128,
        n_heads=8,
        n_kv_heads=8,
        is_causal=True,
        local_window_size=None,
        block_q=512,
        block_kv=512,
        block_kv_compute=512,
        block_q_dkv=512,
        block_kv_dkv=512,
        block_kv_dkv_compute=512,
        block_q_dq=None,
        block_kv_dq=None,
        use_fused_bwd_kernel=True,
        use_vmap_bwd=False,
        use_document_aware_block_skipping=True,
        q_layout="HEAD_DIM_MINOR",
        k_layout="HEAD_DIM_MINOR",
        v_layout="HEAD_DIM_MINOR",
    )
    reversed_splash_args = dict(reversed(list(splash_args.items())))

    fwd1, bwd1 = self._get_kernel_names_in_subprocess(fn, **splash_args)
    fwd2, bwd2 = self._get_kernel_names_in_subprocess(
        fn, **reversed_splash_args
    )

    self.assertEqual(fwd1, fwd2)
    self.assertEqual(bwd1, bwd2)

    no_block_skipping_args = dict(splash_args)
    no_block_skipping_args["use_document_aware_block_skipping"] = False
    fwd3, bwd3 = self._get_kernel_names_in_subprocess(
        fn, **no_block_skipping_args
    )
    self.assertNotEqual(fwd1, fwd3)
    self.assertNotEqual(bwd1, bwd3)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
