import torch
from torch.utils.data import Dataset
import pandas as pd

def data_process(df):
    # Extract RUL for training set
    rul_per_unit = df.groupby(0)[1].max().reset_index()
    rul_per_unit.columns = ["unit", "max_cycle"]
    df = df.rename(columns={0: "unit", 1: "cycle"})
    df = df.merge(rul_per_unit, on="unit", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df

class ServiceData(Dataset):
    def __init__(self, df, seq_len=30):
        """
        df       : preprocessed dataframe with ['unit','cycle','RUL', features...]
        seq_len  : sequence length for time series
        unit_ids : list of engine IDs to include (if None → use all units)
        """
        self.df = df.copy().reset_index(drop=True)

        self.seq_len = seq_len

        # Features = all numeric columns except meta/labels
        self.features = self.df.columns.difference(
            ["unit", "cycle", "max_cycle", "RUL"]
        ).tolist()

        # Build all (X,y) sequences
        self.samples = []
        grouped = self.df.groupby("unit")
        for _, group in grouped:
            group = group.sort_values("cycle")
            values = group[self.features].values
            rul_values = group["RUL"].values

            for i in range(len(group) - seq_len + 1):
                x_seq = values[i:i+seq_len]
                y_val = rul_values[i+seq_len-1]
                self.samples.append((x_seq, y_val))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)