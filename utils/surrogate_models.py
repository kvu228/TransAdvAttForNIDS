"""Mô hình **surrogate** (60 đặc trưng) phục vụ sinh gradient cho tấn công.

Gồm MLP, CNN 1D, ResCNN, LSTM (reshape 12×5), Self-Attention (6×10 + positional encoding).
Tên class kết thúc bằng ``_s``; khớp ``fea_s.csv`` và minmax ``*_minmax_s``.
"""
import torch
import torch.nn as nn
import math

class mlp_s(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(60, 256), # layer 1
            nn.ReLU(),
            
            nn.Linear(256, 256), # layer 2
            nn.ReLU(),

            nn.Linear(256, 256), # layer 3
            nn.ReLU(),

            nn.Linear(256, 256), # layer 4
            nn.ReLU(),

            nn.Linear(256, 2) # classifier
        )

    def forward(self, x):
        return self.model(x)

class lstm_s(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 256, 3, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 2)
        )
    
    def forward(self, x):
        x, _ = self.lstm(torch.reshape(x, (x.size(0), 12, 5)))
        return self.classifier(x)

class con_norm_relu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(con_norm_relu, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 5, 1, "same"),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class cnn_s(nn.Module):
    def __init__(self):
        super(cnn_s, self).__init__()
        self.fisrt_layer = con_norm_relu(1, 64)
        self.middle_layers = nn.ModuleList([con_norm_relu(64, 64) for _ in range(5)])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3840, 2),
        )

    def forward(self, x):
        x = self.fisrt_layer(x.unsqueeze(1))
        for layer in self.middle_layers:
            x = layer(x)
        return self.classifier(x)

class ResCNN_s(nn.Module):
    def __init__(self):
        super(ResCNN_s, self).__init__()
        self.layer1 = con_norm_relu(1, 64)
        self.cnn_layers = nn.ModuleList([con_norm_relu(64, 64) for _ in range(5)])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3840, 2)
        )

    def forward(self, x):
        x = self.layer1(x.unsqueeze(1))
        for layer in self.cnn_layers:
            x = layer(x) + x
        return self.classifier(x)

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size, output_size):
        super(SelfAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, output_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_size).float())
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention = torch.matmul(attention_weights, value)

        return self.fc_out(attention)

class SelfAttention_s(nn.Module):
    def __init__(self, embed_size=10):
        super(SelfAttention_s, self).__init__()
        self.selfAttentionLayer = SelfAttentionLayer(embed_size=embed_size, output_size=embed_size)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 2)
        )
        self.positional_encoding = self._get_positional_encoding(10)
    
    def _get_positional_encoding(self, hidden_size, max_len=10):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        position_encoding = torch.zeros(max_len, hidden_size)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding
    
    def forward(self, x):

        x = torch.reshape(x, (x.size(0), 6, 10))

        _, seq_len, _ = x.size()
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0).to(x.device)

        x = self.selfAttentionLayer(x)
        x = self.mlp(x)
        return x