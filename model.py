# model.py
import torch
import torch.nn as nn

class SnoreNet(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=1):
        super(SnoreNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):  # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.cnn(x)       # [B, C, T]
        x = x.transpose(1, 2) # [B, T, C]
        x, _ = self.lstm(x)   # [B, T, 2H]
        x = self.fc(x)        # [B, T, 1]
        return x
