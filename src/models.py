"""
models.py -- lightweight sequence-model zoo for the ablation/comparison study.

  - Spatial LSTM / Spatial+Angular LSTM: your original Section-5 design
    (single LSTM layer, 64 hidden units, small FC head).
  - GRU: same shape, GRU cell instead of LSTM -- cited in your related-work
    table (Guo et al. 2025 uses GRU-hybrid baselines), included so "we chose
    LSTM" is a demonstrated choice, not an assumption.
  - Bi-LSTM: cited directly in your synopsis (Yang [33], ref [7] Yao et al.) as
    the established architecture for this exact task -- a paper needs this
    comparison point.

All three stay in the "lightweight" scope from Section 8: one recurrent layer,
no CNN backbone, no attention/Transformer.
"""
import torch
import torch.nn as nn


class LightweightRNN(nn.Module):
    def __init__(self, in_features, horizon, hidden=64, cell="lstm", bidirectional=False):
        super().__init__()
        cell = cell.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(in_features, hidden, num_layers=1, batch_first=True,
                            bidirectional=bidirectional)
        self.cell = cell
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, horizon * 2),
        )
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # handles both uni- and bi-directional cleanly
        pred = self.head(last)
        return pred.view(-1, self.horizon, 2)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(name, in_features, horizon, hidden=64):
    """name in {lstm, gru, bilstm}"""
    if name == "lstm":
        return LightweightRNN(in_features, horizon, hidden, cell="lstm", bidirectional=False)
    if name == "gru":
        return LightweightRNN(in_features, horizon, hidden, cell="gru", bidirectional=False)
    if name == "bilstm":
        return LightweightRNN(in_features, horizon, hidden, cell="lstm", bidirectional=True)
    raise ValueError(name)


# Backward-compat alias so existing imports (`from models import LightweightLSTM`) still work.
class LightweightLSTM(LightweightRNN):
    def __init__(self, in_features, horizon, hidden=64):
        super().__init__(in_features, horizon, hidden, cell="lstm", bidirectional=False)
