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
from models.rnn import RNNRUL
from models.lstm import LSTMRUL
from models.seq2seq import Seq2SeqRUL
from models.informer import InformerRUL, InformerEncoder
import joblib
from torch.utils.data import DataLoader, Dataset
from service.utils import ServiceData, data_process

# Load parameters
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)
seq_len = model.seq_len
batch_size = model.batch_size
device = model.device

# Process data
cols = list(range(26))  # 26 columns
df = pd.read_csv(os.path.join("service/", "test_api_data.txt"), sep=r"\s+", header=None, names=cols)
df = data_process(df)
# Run prediction
pred_dataset  = ServiceData(df, seq_len=seq_len)
pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
print(f"Prediction dataset: {pred_dataset}")
model.eval()
preds = []

model.to(device)
with torch.no_grad():
    for x_pred, __ in pred_loader:
        print(f"X dataset: {x_pred}")
        x_pred = x_pred.to(device)
        output = model(x_pred)
        print(f"output dataset: {output}")
        preds.extend(output.cpu().numpy().ravel())
print(f"Predicted RUL: {preds}")