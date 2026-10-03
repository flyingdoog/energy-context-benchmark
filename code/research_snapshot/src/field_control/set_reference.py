"""Small Deep Sets-style group regressor, adapted to ten native power bins.

Architecture source: Zaheer et al., NeurIPS 2017, section 3.1.
The width, mean pooling plus size, and forecasting head are a local recipe,
not a claim to reproduce an original-paper forecasting experiment.
"""
import torch
from torch import nn


class SetPowerRegressor(nn.Module):
    def __init__(self, input_width, hidden=64, leads=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_width, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(),
                                  nn.Linear(hidden, leads))

    def forward(self, values, segments, counts):
        encoded = self.encoder(values)
        pooled = encoded.new_zeros((len(counts), encoded.shape[1]))
        pooled.index_add_(0, segments, encoded)
        counts = counts.to(dtype=encoded.dtype)
        pooled = pooled / counts[:, None]
        return self.head(torch.cat((pooled, torch.log1p(counts)[:, None]), dim=1))


def tensor_batch(inputs, normalizer, indices, *, reassigned=False):
    values, segments, counts = inputs.batch(indices, reassigned=reassigned)
    return (torch.from_numpy(normalizer.transform(values)),
            torch.from_numpy(segments), torch.from_numpy(counts))
