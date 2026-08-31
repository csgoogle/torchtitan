# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for the scan test suite.

A decoder-block stand-in with the real call signature but a fraction of the
cost, the model that owns them, and the three operations every scan test starts
with: stack the layers, shard the stack, trace it. Every scan test builds on
these, so the thing under test is the stack, not a per-file re-invention of a
toy model.
"""

import collections

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.graph_trainer.simple_fsdp import MixedPrecisionPolicy
from torchtitan.experiments.tpu.scan import fsdp as scan_fsdp
from torchtitan.experiments.tpu.scan import layer_stack


# The toy world: small enough to compare gradients exactly, deep enough that
# "one traced body regardless of depth" is a claim with content.
N_LAYERS, DIM, BATCH, SEQ = 4, 8, 2, 3


class ToyBlock(nn.Module):
  """A `TransformerBlock` stand-in with llama3's block signature.

  Matching the real signature -- hidden state first, then the loop-invariant
  arguments `Decoder.forward` passes every layer -- is what makes the stack
  exercise its actual `functional_call` path rather than a simplified one.
  """

  def __init__(self, dim: int):
    super().__init__()
    self.wi = nn.Linear(dim, dim, bias=False)
    self.wo = nn.Linear(dim, dim, bias=False)

  def forward(self, h, attention_masks=None, positions=None, segment_ids=None):
    del attention_masks, positions, segment_ids
    return h + self.wo(torch.tanh(self.wi(h)))


class ToyModel(nn.Module):
  """Owns a `ModuleDict` of blocks keyed by index, as `Decoder` does."""

  def __init__(self, n_layers: int = N_LAYERS, dim: int = DIM):
    super().__init__()
    self.layers = nn.ModuleDict({str(i): ToyBlock(dim) for i in range(n_layers)})

  def forward(self, h, attention_masks=None, positions=None, segment_ids=None):
    # Iterating `.values()` is what `Decoder.forward` does, and it is the only
    # thing `ScannedLayerStack` promises to stay compatible with: it yields
    # `[self]`, so this loop runs once and the single "layer" is the whole scan.
    for layer in self.layers.values():
      h = layer(h, attention_masks, positions, segment_ids)
    return h


def unrolled_reference(model: ToyModel, h: torch.Tensor) -> torch.Tensor:
  """Runs the blocks in a plain Python loop -- the semantics scan must preserve."""
  for i in range(len(model.layers)):
    h = model.layers[str(i)](h)
  return h


def stacked(device, n_layers: int = N_LAYERS, dim: int = DIM, **stack_kwargs):
  """`(model, stack)` for a model whose layers have collapsed into one scan.

  Seeded, so an independently built `ToyModel(n_layers, dim)` under the same
  seed holds exactly the weights the stack was built from. `stack_kwargs` go
  through to `stack_layers` (`recompute_body`, `peel_first_layer`).
  """
  torch.manual_seed(0)
  model = ToyModel(n_layers, dim).to(device)
  return model, layer_stack.stack_layers(model, **stack_kwargs)


def shard(stack, device, world_size, param_dtype=None, reduce_dtype=None):
  """Puts `stack` under SimpleFSDP on a 1-D mesh; returns the gather plan."""
  return scan_fsdp.shard_stacked_layers(
      stack,
      dp_mesh=init_device_mesh(device.type, (world_size,)),
      dp_mode="fully_shard",
      mp_policy=MixedPrecisionPolicy(
          param_dtype=param_dtype, reduce_dtype=reduce_dtype
      ),
  )


def trace(module, h):
  """`(root graph, [body graphs])` for `module` traced at `h`.

  "real", not "fake": the stack holds real parameters, and tracing the HOP under
  a second FakeTensorMode trips "Mixing fake modes NYI".
  """
  gm = make_fx(module, tracing_mode="real")(h)
  bodies = [s.graph for n, s in gm.named_modules() if n and hasattr(s, "graph")]
  return gm.graph, bodies


def collectives(graph) -> collections.Counter:
  """Counts the `_c10d_functional` ops in an fx graph, keyed by op name."""
  return collections.Counter(
      node.target._schema.name.split("::")[-1]  # pylint: disable=protected-access
      for node in graph.nodes
      if isinstance(node.target, torch._ops.OpOverload)
      and node.target.namespace == "_c10d_functional"
  )
