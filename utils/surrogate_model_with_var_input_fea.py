"""MLP surrogate với số chiều đầu vào **tùy chỉnh** (``fea_num``).

Dùng trong thí nghiệm 5.4.4 (quét số đặc trưng 27…66); kiến trúc giống ``mlp_s`` nhưng
``Linear(fea_num, 256)`` thay cho cố định 60.
"""
import torch
import torch.nn as nn


class mlp_s_varfea(nn.Module):
    """MLP 4 tầng ẩn 256 + phân loại nhị phân (logits 2 lớp)."""

    def __init__(self, fea_num):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(fea_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    net = mlp_s_varfea(80)
    x = torch.rand((128, 80))
    y = net(x)
    print(y.shape)
