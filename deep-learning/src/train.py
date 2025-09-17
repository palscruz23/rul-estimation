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

# Setup ML flow
print("Setting up MLFlow...\n")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("predictive-rul-deep-learning")

print("Loading data...\n")
train_df = data_process()
feature_cols = train_df.columns.difference(["unit", "cycle", "max_cycle", "RUL"])
num_features=len(train_df[feature_cols].columns)

models = {
    "RNN": RNNRUL(num_features),
    "LSTM": LSTMRUL(num_features),
    "Seq2Seq": Seq2SeqRUL(num_features),
    "Informer": InformerRUL(num_features)   
}

# Perform grid search with MLFlow
model_results = {}
model_names = ["Informer", "RNN", "LSTM", "Seq2Seq"]   # <-- change to "RNN", "LSTM", "Seq2Seq", "Informer"
device="cuda" if torch.cuda.is_available() else "cpu"

# Params for grid search
print("Performing grid search on different models...")
for model_name in model_names:
    model = models[model_name].to(device)
    if model_name == "Informer":
            param_grid = {
            "opt": ["rms"], 
            "seq_len": [100],
            "batch_size": [32],
            "lr": [1e-3],
            "d_model": [64],
            "n_heads": [2],
            "num_layers": [1],
            "epochs": [1],
            "dropout": [0.2]
            }
    else:
            param_grid = {
            "opt": ["rms"], 
            "hidden_dim1": [128],
            "hidden_dim2": [64],
            "num_layers": [1],
            "dropout": [0.2],
            "seq_len": [100],
            "batch_size": [32],
            "lr": [1e-3],
            "epochs": [1]  
            }
    best_model, best_params, best_loss, results = grid_search(
        model.__class__, param_grid,
        train_df, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model_results[model.__class__.__name__] =  {
                                                    "Best Model": best_model,
                                                    "Best Params": best_params,
                                                    "Best Loss": best_loss,
                                                    "results": results
                                                }
print("Check performance in MLFlow UI...\n")
print("""Run in terminal `mlflow ui --port 5000`
"Open `http://127.0.0.1:5000/` in browser.""")



