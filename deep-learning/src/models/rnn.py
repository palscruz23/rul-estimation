import torch as torch
import torch.nn as nn

class RNNRUL(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=56, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim1, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)
        self.rnn2 = nn.RNN(hidden_dim1, hidden_dim2, num_layers, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)           # [B, L, H]
        out = out[:, -1, :]            # take last timestep
        out = self.bn(out)             # batch norm
        out = self.dropout1(out)       # first dropout

        out, _ = self.rnn2(out.unsqueeze(1))  # add 2nd RNN
        out = out[:, -1, :]

        out = self.dropout2(out)
        out = self.fc(out)
        return self.relu(out).squeeze(-1)
    
