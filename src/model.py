"""Character-level recurrent neural network for spam detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpamRNN(nn.Module):
    """Character-level GRU classifier.

    Characters are represented as one-hot vectors. The GRU outputs are pooled
    using both max pooling and average pooling, then classified with a small MLP.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len), values are character IDs
        one_hot = F.one_hot(x, num_classes=self.vocab_size).float()
        out, _ = self.rnn(one_hot)

        max_pool = torch.max(out, dim=1)[0]
        avg_pool = torch.mean(out, dim=1)
        pooled = torch.cat([max_pool, avg_pool], dim=1)

        hidden = F.relu(self.fc1(pooled))
        return self.fc2(hidden)
