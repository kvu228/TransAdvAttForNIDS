"""Biến thể target **78 đặc trưng** (cùng họ kiến trúc với ``target_models``).

Dùng khi ``load_net(78, ...)`` hoặc thí nghiệm mở rộng không gian đặc trưng (vd. mục 5.2).
"""
import torch.nn as nn
import torch
import math
import torch.nn.init as init

class mlp_t_78(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(78, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class lstm_t_78(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 256, 3, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2)
        )
    
    def forward(self, x):
        pad = torch.zeros(x.size(0), 2, device=x.device, dtype=x.dtype)
        x = torch.cat((x, pad), dim=1)
        x, _ = self.lstm(torch.reshape(x, (x.size(0), 16, 5)))
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

class cnn_t_78(nn.Module):
    def __init__(self):
        super().__init__()
        self.fisrt_layer = con_norm_relu(1, 64)
        self.middle_layers = nn.ModuleList([con_norm_relu(64, 64) for _ in range(5)])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4992, 2),
        )
    
    def forward(self, x):
        x = self.fisrt_layer(x.unsqueeze(1))
        for layer in self.middle_layers:
            x = layer(x)
        return self.classifier(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class ResCNN_t_78(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = con_norm_relu(1, 64)
        self.cnn_layers = nn.ModuleList([con_norm_relu(64, 64) for _ in range(5)])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4992, 2)
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

class SelfAttention_t_78(nn.Module):
    def __init__(self, embed_size=10):
        super().__init__()
        self.selfAttentionLayer = SelfAttentionLayer(embed_size=embed_size, output_size=embed_size)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80, 100),
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
        pad = torch.zeros(x.size(0), 2, device=x.device, dtype=x.dtype)
        x = torch.cat((x, pad), dim=1)
        x = torch.reshape(x, (x.size(0), 8, 10))

        _, seq_len, _ = x.size()
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0).to(x.device)
        
        x = self.selfAttentionLayer(x)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    net = SelfAttention_t_78()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev)
    x = torch.rand((128, 78), device=dev)
    y = net(x)
    print(y.shape)