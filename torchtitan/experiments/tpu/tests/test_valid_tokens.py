# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed.distributed_c10d as c10d

from torchtitan.experiments.tpu.valid_tokens import (
    loss_with_global_valid_token_sum,
)
from torchtitan.trainer import Trainer


class ValidTokenReductionTest(absltest.TestCase):
    def test_default_trainer_path_keeps_eager_reduction(self) -> None:
        trainer = object.__new__(Trainer)
        trainer.parallel_dims = SimpleNamespace(
            dp_enabled=True,
            get_mesh=lambda name: f"{name}-mesh",
        )
        trainer.device = torch.device("cpu")
        local_valid_tokens = torch.tensor(3)

        with mock.patch(
            "torchtitan.trainer.dist_utils.dist_sum", return_value=12.0
        ) as dist_sum:
            prepared = trainer._prepare_valid_tokens_for_loss(local_valid_tokens)

        self.assertEqual(prepared, 12.0)
        dist_sum.assert_called_once()
        self.assertEqual(dist_sum.call_args.args[1], "batch-mesh")
        self.assertEqual(
            trainer._materialize_global_valid_tokens_for_metrics(
                local_valid_tokens, prepared
            ),
            12.0,
        )

    def test_reduces_valid_tokens_after_forward_compute(self) -> None:
        calls = []

        def all_reduce(value, *, reduceOp, group):
            calls.append((value.clone(), reduceOp, group))
            return value * 4

        def loss_fn(pred, labels, *, global_valid_tokens):
            del labels
            return pred.sum() / global_valid_tokens

        wrapped = loss_with_global_valid_token_sum(
            loss_fn,
            "batch-mesh",
            all_reduce=all_reduce,
        )

        def step(inputs, labels, valid_tokens):
            pred = torch.sin(inputs)
            return wrapped(
                pred,
                labels,
                global_valid_tokens=valid_tokens,
            )

        torch.testing.assert_close(
            step(torch.tensor([8.0]), torch.tensor([0]), torch.tensor(2)),
            torch.sin(torch.tensor([8.0])).sum() / 8,
        )
        self.assertLen(calls, 1)
        self.assertEqual(calls[0][1], c10d.ReduceOp.SUM.name)
        self.assertEqual(calls[0][2], "batch-mesh")

        graph = torch.fx.symbolic_trace(step).graph
        targets = [node.target for node in graph.nodes if node.op == "call_function"]
        self.assertLess(targets.index(torch.sin), targets.index(operator.mul))


if __name__ == "__main__":
    absltest.main()
