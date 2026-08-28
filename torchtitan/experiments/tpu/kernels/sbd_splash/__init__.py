# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored tokamax splash attention (pin 94ac46a) -- splash-bias-dropout family fork.

Vendored per the family binding (capability-eval, home_repo none): self-contained,
re-gradable. Upstream: raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/.
Fork surface: bias + dropout additions per the v001 stub's Mechanism.
"""
from torchtitan.experiments.tpu.kernels.sbd_splash.splash_attention_kernel import *  # noqa: F401,F403
