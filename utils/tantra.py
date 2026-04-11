"""Baseline TANTRA: LSTM many-to-one trên chuỗi đặc trưng (paper Sec. IV-C, Fig. 2).

``ws=150`` → ``seq_len=151``; ``feat_dim=2`` mặc định. Dùng so sánh mục 5.3.2 (Table 9).
"""
import torch.nn as nn

class TantraLSTM(nn.Module):
    """LSTM + MLP hồi quy; output scalar (logit/risk) từ hidden cuối."""
    def __init__(self, ws: int = 150, feat_dim: int = 2):
        super().__init__()
        self.seq_len  = ws + 1
        self.feat_dim = feat_dim

        self.lstm  = nn.LSTM(input_size=feat_dim,
                             hidden_size=32,
                             num_layers=1,
                             batch_first=True)

        self.reg   = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        _, (h_last, _) = self.lstm(x)   # h_last : (1 , B , 32)
        h = h_last.squeeze(0)           # (B , 32)
        y = self.reg(h)                 # (B , 1)
        return y