"""Tests for Conformer model instantiation and forward pass."""

from absl.testing import absltest
import torch
import torchtitan.experiments.torchax.conformer as conformer


class ConformerModelTest(absltest.TestCase):

  def test_instantiation_and_forward(self):
    args = conformer.args["debugmodel"]

    conformer_model_inst = conformer.model(args)

    batch_size = 2
    seq_len = 16
    inputs = torch.randint(0, args.vocab_size, (batch_size, seq_len), dtype=torch.long)

    logits = conformer_model_inst(inputs)

    self.assertEqual(logits.shape, (batch_size, seq_len, args.vocab_size))
    print("Forward pass successful. Shape:", logits.shape)


if __name__ == "__main__":
  absltest.main()
