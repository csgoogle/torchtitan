import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedGroupCausalConv(nn.Module):
    """
    TorchTPU / XLA Optimized Dilated Grouped Causal Conv.
    This optimized version materializes exactly ONE padded memory block for all groups 
    and uses zero-copy shifted XLA slices to feed the convolutions perfectly aligned vectors.
    """
    def __init__(self, d_model=64, group_size=4, kernel_size=4):
        super().__init__()
        assert d_model % group_size == 0, 'group_size must divide d_model'
        self.gs, self.ng, self.K = group_size, d_model // group_size, kernel_size

        # Calculate the absolute max boundary padding we will ever need across all groups.
        self.max_dilation = 2**(self.ng - 1)
        self.max_pad = (self.K - 1) * self.max_dilation

        self.convs = nn.ModuleList([
            nn.Conv1d(group_size, group_size, kernel_size,
                      groups=group_size, dilation=2**i, bias=True)
            for i in range(self.ng)
        ])

    def forward(self, x):
        # Transpose input: [B, T, D] -> [B, D, T]
        # Required because PyTorch Conv1d expects spatial dimension last (Channels, Length)
        # while standard transformer block interfaces pass (Length, d_model).
        h = x.transpose(1, 2)
        T = x.shape[1]

        h_padded = F.pad(h, (self.max_pad, 0))
        outs = []
        for i, conv in enumerate(self.convs):
            # The causal pad needed for this specific dilation level
            pad_i = (self.K - 1) * (2**i)
            # Extract a zero-copy shifted view from the single padded block.
            start_idx = self.max_pad - pad_i

            sl_view = h_padded[:, i*self.gs:(i+1)*self.gs, start_idx : start_idx + T + pad_i]
            outs.append(conv(sl_view))

        return torch.cat(outs, dim=1).transpose(1, 2)
