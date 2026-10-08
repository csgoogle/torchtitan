# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Names of the regions that TPU selective memory policies can save.

The names follow upstream ``RegionAC`` regions (``<block-relative module
FQN>.<local name>``), so eager and graph AC can later share one vocabulary.
``--tpu-config.graph-trainer-save-regions`` takes these names.

Model code marks values that op names alone cannot tell apart with
``torch.fx.traceback.annotate_fn({REMAT_REGION: <region name>})``.

This module imports nothing, so model code can use these names without
importing the graph trainer.
"""

# The ``annotate_fn`` key whose value names a region.
REMAT_REGION = "remat_region"

# The Splash output and LSE that the attention backward reads.
ATTENTION_INNER_ATTENTION = "attention.inner_attention"
# The expert-sort permutation in ``token_dispatcher.local_reorder``.
MOE_ROUTE_PERMUTATION = "moe.route_permutation"
