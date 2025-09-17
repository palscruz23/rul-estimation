import torch as torch
import torch.nn as nn

class Seq2SeqRUL(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=56, num_layers=2, dropout=0.3, epochs=10, seq_len=100, batch_size=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_layers = num_layers    
        self.dropout = dropout
        self.epochs = epochs
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Encoder
        self.encoder1 = nn.LSTM(input_dim, hidden_dim1, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)

        # Second Encoder-like block
        self.encoder2 = nn.LSTM(hidden_dim1, hidden_dim2, num_layers, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim2, 1)


    def forward(self, x):
        enc_out, _ = self.encoder1(x)
        out = enc_out[:, -1, :]        # last timestep
        out = self.bn(out)
        out = self.dropout1(out)

        out, _ = self.encoder2(out.unsqueeze(1))
        out = out[:, -1, :]

        out = self.dropout2(out)
        out = self.fc(out)
        return self.relu(out).squeeze(-1)