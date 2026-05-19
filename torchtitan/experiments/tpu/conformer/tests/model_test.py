from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.conformer import model


class ConformerModelTest(absltest.TestCase):

  def test_instantiation_and_forward(self):
    args = model.ConformerModelArgs(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        kernel_size=31,
    )

    conformer_model = model.Conformer(args)

    batch_size = 2
    seq_len = 16
    inputs = torch.randint(0, args.vocab_size, (batch_size, seq_len), dtype=torch.long)

    logits = conformer_model(inputs)

    self.assertEqual(logits.shape, (batch_size, seq_len, args.vocab_size))
    print("Forward pass successful. Shape:", logits.shape)


if __name__ == "__main__":
  absltest.main()
