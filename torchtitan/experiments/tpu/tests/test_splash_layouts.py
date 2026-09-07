# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every llama3 recipe that turns splash on must say how it wants Q and K laid
out.

Left undeclared, splash defaults both to `HEAD_DIM_MINOR` and XLA inserts the
transposes. Unrolled that is a 0.06 s retile, because backward layout inference
reaches the producer. Under `--tpu-config.use-scan-layers` the producer sits
inside a `while` body, where its output layout is pinned to the loop's, so the
inference cannot reach it and the same transpose costs 3.35 s. Declaring the
layouts is what let the 70B scan arm hold 43.59 % MFU instead of 42.73 %.

The measured effect was on the exposed all-gather (1.71 -> 0.41 s), not on
splash itself (38.09 -> 38.13 s): the layout changed what could overlap, not
what the kernel costs.
"""

import unittest

from torchtitan.experiments.tpu.llama3 import config_registry


class TestSplashLayoutsAreDeclared(unittest.TestCase):
  def test_every_splash_recipe_declares_q_and_k(self):
    missing = []
    for name in dir(config_registry):
      if name.startswith("_"):
        continue
      fn = getattr(config_registry, name)
      if not callable(fn) or getattr(fn, "__module__", None) != (
          config_registry.__name__
      ):
        continue
      try:
        cfg = fn()
      except TypeError:
        continue  # not a zero-argument recipe
      sa = getattr(cfg, "splash_attention_kernel", None)
      if sa is None or not sa.use_splash_attention_kernel:
        continue
      if sa.sa_q_layout != "SEQ_MINOR" or sa.sa_k_layout != "SEQ_MINOR":
        missing.append((name, sa.sa_q_layout, sa.sa_k_layout))
    self.assertEqual(
        missing,
        [],
        "these recipes enable splash without declaring SEQ_MINOR q/k, which "
        "costs 3.3 s a step under scan: " + repr(missing),
    )


if __name__ == "__main__":
  unittest.main()
