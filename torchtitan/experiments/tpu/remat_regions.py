# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Names of the regions that TPU selective memory policies can save.

The names follow upstream ``RegionAC`` regions (``<block-relative module
FQN>.<local name>``), so eager and graph AC can later share one vocabulary.
``--tpu-config.graph-trainer-save-regions`` takes these names.

Model code marks values that op names alone cannot tell apart by computing
them inside ``with annotate_region(<region name>):``.

This module imports only torch, so model code can use it without importing
the graph trainer.
"""

from contextlib import AbstractContextManager

from torch.fx.traceback import annotate

# The FX node annotation key whose value names a region.
REMAT_REGION = "remat_region"

# The Splash output and LSE that the attention backward reads.
ATTENTION_INNER_ATTENTION = "attention.inner_attention"
# The expert-sort permutation in ``token_dispatcher.local_reorder``.
MOE_ROUTE_PERMUTATION = "moe.route_permutation"
# The router's top-k expert ids: upstream's ``routing_decision`` region in
# ``TokenChoiceTopKRouter``.
MOE_ROUTING_DECISION = "moe.router.routing_decision"


def annotate_region(name: str) -> AbstractContextManager[None]:
  """Mark the FX nodes traced inside this ``with`` block as region ``name``.

  This only adds node metadata (``node.meta["custom"][REMAT_REGION]``); it does
  not change what runs.
  """
  return annotate({REMAT_REGION: name})
