import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
import mlflow
from utils.mlflow_helper import start_mlflow_run, log_epoch_metrics
from utils.helper import *
from models.rnn import RNNRUL
from models.lstm import LSTMRUL
from models.seq2seq import Seq2SeqRUL
from models.informer import InformerRUL, InformerEncoder
import joblib
from torch.utils.data import DataLoader

# Load parameters
MODEL_PATH = "src/models/model.pkl"
model = joblib.load(MODEL_PATH)
seq_len = model.seq_len
batch_size = model.batch_size
device = model.device

train_df = data_process()
test_df, test_units = data_split(train_df, seq_len, batch_size, test_size=0.1, random_state=42, test=1)

test_dataset  = PHM08RULDataset(test_df, seq_len=seq_len, unit_ids=test_units)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
preds = []
true_rul = []
test_loss = 0

model.to(device)
criterion = nn.MSELoss()
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        output = model(x_test)
        preds.extend(output.cpu().numpy().ravel())
        true_rul.extend(y_test.cpu().numpy().ravel())
        test_loss += criterion(output, y_test).item()
avg_test_loss = test_loss / len(test_loader)

print("="*60)
print(f"Test Set Evaluation:")
print(f"MSE: {avg_test_loss:.4f}")
print("="*60)
print("Plotting test RUL curves...")
plot_test_results(true_rul, preds, model)