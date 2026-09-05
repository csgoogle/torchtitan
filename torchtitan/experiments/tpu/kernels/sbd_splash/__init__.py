# Copyright 2026 Google LLC

"""Vendored tokamax splash attention (pin 94ac46a) -- splash-bias-dropout family fork.

Vendored per the family binding (capability-eval, home_repo none): self-contained,
re-gradable. Upstream: raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/.
Fork surface: bias + dropout additions per the v001 stub's Mechanism.
"""
from torchtitan.experiments.tpu.kernels.sbd_splash.splash_attention_kernel import *  # noqa: F401,F403
