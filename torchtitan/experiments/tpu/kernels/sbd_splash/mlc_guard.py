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

"""sbd fork v008: guarded max_logit_const (MLC) adoption helper.

MLC replaces the online running-max with a compile-time constant shift; the
family default is ON where a logit bound is known. The failure mode is
UNDERFLOW: if every logit in a row sits far below the const, 2^(logit-const)
flushes to zero, l == 0, and the output is NaN (division by zero). The
overflow direction needs logit - const > ~127 (base-2) and is bounded by the
same window check.

Rules (v008, load-bearing -- see the guard demonstration gate):
- bound is CALLER-SUPPLIED (bias-tensor path) or ANALYTIC (bias_function with
  a declared slope/offset bound; features-off uses the qk envelope). NEVER
  computed by max-reducing the bias operand (O(S^2) read, host-preprocess
  anti-pattern).
- MLC engages iff the whole plausible logit range sits inside a safe window
  around the const: const - bound_low <= UNDER_MARGIN and
  bound_high - const <= OVER_MARGIN.
"""

UNDER_MARGIN = 80.0   # base-2 exp underflows ~2^-126; 80 leaves headroom
OVER_MARGIN = 60.0    # 2^(+127) overflows; 60 leaves headroom
DEFAULT_CONST = 30.0


def mlc_or_none(bound_high, bound_low=None, const=DEFAULT_CONST):
    """Return `const` if the declared logit bounds make MLC safe, else None
    (running-max fallback). bound_high: max plausible logit; bound_low: max
    logit of the SMALLEST row-max plausible (defaults to -bound_high)."""
    if bound_high is None:
        return None
    if bound_low is None:
        bound_low = -abs(bound_high)
    if bound_high - const > OVER_MARGIN:
        return None
    if const - bound_low > UNDER_MARGIN:
        return None
    return const
