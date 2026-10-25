import torch
import torch.nn as nn


class CNNBiLSTMAttention(nn.Module):
    """Conv1D(64,k=3) -> BiLSTM(64/dir) -> soft attention over time -> dense head."""
    def __init__(self, n_features, conv=64, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, conv, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(conv, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.last_attention = None

    def forward(self, x):                      # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)   # (B, T)
        self.last_attention = a.detach()
        ctx = (a.unsqueeze(-1) * h).sum(1)
        return self.head(ctx).squeeze(-1)
