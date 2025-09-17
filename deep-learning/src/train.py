import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torchinfo import summary
import mlflow
import time
from utils.mlflow_helper import start_mlflow_run, log_epoch_metrics


# Setup ML flow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("predictive-rul-deep-learning")

num_features=len(train_dataset.features)

models = {
    "RNN": RNNRUL(num_features),
    "LSTM": LSTMRUL(num_features),
    "Seq2Seq": Seq2SeqRUL(num_features),
    "Informer": InformerRUL(num_features=len(train_dataset.features), d_model=128, n_heads=4, num_layers=2, dropout=0.1)   
}

# Perform grid search with MLFlow

model_results = {}
model_names = ["Informer", "RNN", "LSTM", "Seq2Seq"]   # <-- change to "RNN", "LSTM", "Seq2Seq", "Informer"
device="cuda" if torch.cuda.is_available() else "cpu"

# Stops MLFlow instance if it exists
mlflow.end_run()

for model_name in model_names:
    model = models[model_name].to(device)
    if model_name == "Informer":
        param_grid = {
        "opt": ["rms", "adam"], 
        "seq_len": [100],
        "batch_size": [32, 64],
        "lr": [1e-3, 1e-2],
        "d_model": [64],
        "n_heads": [2],
        "num_layers": [1],
        "epochs": [10],
        "dropout": [0.2]
        }
    else:
        param_grid = {
        "opt": ["rms", "adam"], 
        "hidden_dim1": [128],
        "hidden_dim2": [64],
        "num_layers": [1, 2],
        "dropout": [0.2],
        "seq_len": [100],
        "batch_size": [32, 64],
        "lr": [1e-3, 1e-2],
        "epochs": [10]  
        }
    best_model, best_params, best_loss, results = grid_search(
        model.__class__, param_grid,
        train_dataset, val_dataset, train_units, val_units, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model_results[model.__class__.__name__] =  {
                                                    "Best Model": best_model,
                                                    "Best Params": best_params,
                                                    "Best Loss": best_loss,
                                                    "results": results
                                                }