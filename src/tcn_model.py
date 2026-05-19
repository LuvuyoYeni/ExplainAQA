import torch
import torch.nn as nn


class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        self.pad  = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=self.pad, dilation=d)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        self.net  = nn.Sequential(
            CausalConv(in_ch, out_ch, k, d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class TCNQualityRegressor(nn.Module):
    def __init__(self, in_features=7, num_channels=64, kernel_size=3):
        super().__init__()
        nc = num_channels
        self.blocks = nn.ModuleList([
            TCNBlock(in_features, nc, kernel_size, 1),
            TCNBlock(nc,          nc, kernel_size, 2),
            TCNBlock(nc,          nc, kernel_size, 4),
            TCNBlock(nc,          nc, kernel_size, 8),
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(nc, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        x = self.blocks[3](x)
        return self.head(x).squeeze(-1)


class LSTMQualityRegressor(nn.Module):
    def __init__(self, in_features=7, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)
