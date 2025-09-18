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

# Setup ML flow
print("Setting up MLFlow...\n")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("predictive-rul-deep-learning")

print("Loading data...\n")
train_df = data_process()

# Enter best params after grid search here
model_name = "Seq2Seq"   # <-- change to "Informer", "RNN", "LSTM", or "Seq2Seq"
seq_len = 100
batch_size = 32
lr = 1e-2
epochs = 1
opt = "adam" # <-- change to "adam", "rms"
device="cuda" if torch.cuda.is_available() else "cpu"

# Sequencing
train_dataset, train_loader, val_loader = data_split(train_df, seq_len, batch_size, test_size=0.1, random_state=42)
num_features=len(train_dataset.features)

models = {
    "RNN": RNNRUL(num_features),
    "LSTM": LSTMRUL(num_features),
    "Seq2Seq": Seq2SeqRUL(num_features),
    "Informer": InformerRUL(num_features)   
}
model = models[model_name]

if model.__class__.__name__ == "InformerRUL":
    params = {
        "num_features": num_features,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 1,
        "dropout": 0.2
        }
    model = model.__class__(**params
    )
    run_params = {
    "model":  model.__class__.__name__,
    "dataset": "PHM08 Dataset",  
    # ---- hyperparams  ---- #
    **params,
    "batch_size": batch_size,
    "seq_len": seq_len,
    "lr": lr,
    "epochs": epochs,
    "opt": opt,
    "device": device
    # -------------------- #
    }

else:
    params = {
        "input_dim": num_features,
        "hidden_dim1": 128,
        "hidden_dim2": 64,
        "num_layers": 1,
        "dropout": 0.2,
        }
    model = model.__class__(**params)
    run_params = {
    "model":  model.__class__.__name__,
    "dataset": "PHM08 Dataset",  
    # ---- hyperparams  ---- #
    **params,
    "batch_size": batch_size,
    "seq_len": seq_len,
    "lr": lr,
    "epochs": epochs,
    "opt": opt,
    "device": device
    # -------------------- #
    }

print(f"Starting training using device: {device}...")
 #Initiate MLFlow run
mlflow_set = 0

if mlflow_set == 1:
    print("Initiate MLFlow logs...")
    run = start_mlflow_run(run_params)
    start_time = time.time()
train_preds, train_true_rul, val_preds, val_true_rul, train_loss, val_loss = train_model(
            model, train_loader, val_loader,
            epochs=epochs,
            lr=lr,
            opt=opt,
            device=device,
            mlflow=mlflow_set
        )
if mlflow_set == 1:
    train_secs = time.time() - start_time
    mlflow.log_metric("train_time_sec", train_secs)
    mlflow.end_run()
print("Inference of best model completed...")

# Save model
joblib.dump(model, "model/model.pkl")
print("Model saved to model/model.pkl")

summary(model, input_size=(batch_size, seq_len, num_features))

print("Plotting loss and validation RUL curves...")
plot_loss_curves(train_loss, val_loss, epochs, val_true_rul, val_preds)
print("Plotting bias and variance...")
plot_bias_variance(train_loss, val_loss, epochs)


