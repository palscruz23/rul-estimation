import pandas as pd
import os
import torch
import mlflow
from utils.mlflow_helper import start_mlflow_run, log_epoch_metrics
from itertools import product
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch.nn as nn
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import joblib

def load_phm08(path, filename):
    cols = list(range(26))  # 26 columns
    df = pd.read_csv(os.path.join(path, filename), sep=r"\s+", header=None, names=cols)
    return df

def data_process():
    train_df = load_phm08("src/data/Challenge_Data/", "train.txt")
    test_df = load_phm08("src/data/Challenge_Data/", "test.txt")

    # Extract RUL for training set
    rul_per_unit = train_df.groupby(0)[1].max().reset_index()
    rul_per_unit.columns = ["unit", "max_cycle"]
    train_df = train_df.rename(columns={0: "unit", 1: "cycle"})
    train_df = train_df.merge(rul_per_unit, on="unit", how="left")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    return train_df

class PHM08RULDataset(Dataset):
    def __init__(self, df, seq_len=30, unit_ids=None):
        """
        df       : preprocessed dataframe with ['unit','cycle','RUL', features...]
        seq_len  : sequence length for time series
        unit_ids : list of engine IDs to include (if None → use all units)
        """
        if unit_ids is not None:
            self.df = df[df["unit"].isin(unit_ids)].reset_index(drop=True)
        else:
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
    
def data_split(train_df, seq_len, batch_size, test_size=0.1, random_state=42, test=0):
    units = train_df["unit"].unique()   # all engine IDs
    training_units, test_units = train_test_split(units, test_size=test_size, random_state=random_state)
    train_units, val_units = train_test_split(training_units, test_size=test_size, random_state=random_state)

    # Apply Feature scaling to transform the scaling of different features into similar scales.
    feature_cols = train_df.columns.difference(["unit", "cycle", "max_cycle", "RUL"])
    training_df = train_df[train_df["unit"].isin(train_units)].copy()
    val_df   = train_df[train_df["unit"].isin(val_units)].copy()
    test_df   = train_df[train_df["unit"].isin(test_units)].copy()

    scaler = StandardScaler()
    training_df[feature_cols] = scaler.fit_transform(training_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    joblib.dump(scaler, "models/scaler.joblib")

    train_dataset = PHM08RULDataset(training_df, seq_len=seq_len, unit_ids=train_units)
    val_dataset   = PHM08RULDataset(val_df, seq_len=seq_len, unit_ids=val_units)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    if test==1:
        return test_df, test_units
    else:
        return train_dataset, train_loader, val_loader
    
def train_model(model, train_loader, val_loader, opt="adam", device="cpu", epochs=20, lr=1e-3, mlflow=0):   
    model.to(device)
    criterion = nn.MSELoss()
    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs+1):
        train_preds = []
        train_true_rul = []
        val_preds = []
        val_true_rul = []

        start = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            train_preds.extend(pred.cpu().detach().numpy().ravel())
            train_true_rul.extend(y.cpu().detach().numpy().ravel())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # # Print every 50 batches
            # if batch_idx % 50 == 0 or batch_idx == len(train_loader):
            #     print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_preds.extend(pred.cpu().numpy().ravel())
                val_true_rul.extend(y.cpu().numpy().ravel())
                val_loss += criterion(pred, y).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        end = time.time()
        total_time = end-start
        print(f"Epoch {epoch} Completed — Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Time Elapsed: {total_time:.2f} seconds")
        if mlflow==1:
            log_epoch_metrics(epoch, avg_train_loss, avg_val_loss)
    print("-"*60)

    return train_preds, train_true_rul, val_preds, val_true_rul, train_losses, val_losses


def grid_search(model_class, param_grid, train_df, device="cuda"):
    results = []
    best_model = None
    best_loss = float("inf")

    # Build train/val loaders
    for params in product(*param_grid.values()):
        hp = dict(zip(param_grid.keys(), params))

        # Sequencing
        seq_len = hp.get("seq_len",10)
        train_dataset, train_loader, val_loader = data_split(train_df, seq_len, hp.get("batch_size",64), test_size=0.1, random_state=42)

        # Model
        if model_class.__name__ == "InformerRUL":
            model = model_class(
                num_features=len(train_dataset.features),
                d_model=hp.get("d_model", 128),
                n_heads=hp.get("n_heads", 4),
                num_layers=hp.get("num_layers", 2)
            )
            run_params = {
            "model": model_class.__name__,
            "dataset": "PHM08 Dataset",  
            # ---- hyperparams  ---- #
            "d_model": hp.get("d_model", 64),
            "n_heads": hp.get("n_heads", 1),
            "num_layers": hp.get("num_layers", 1),
            "batch_size": hp.get("batch_size",64),
            "dropout": hp.get("dropout", 0.1),
            "seq_len": hp.get("seq_len",10),
            "lr": hp.get("lr", 1e-3),
            "epochs": hp.get("epochs", 20),
            "opt": hp.get("opt", "adam"),
            "device": device
            # -------------------- #
            }

        else:
            model = model_class(
                input_dim=len(train_dataset.features),
                hidden_dim1=hp.get("hidden_dim1", 128),
                hidden_dim2=hp.get("hidden_dim2", 128),
                num_layers=hp.get("num_layers", 1),
                dropout=hp.get("dropout", 0.1)
            )
            run_params = {
            "model": model_class.__name__,
            "dataset": "PHM08 Dataset",  
            # ---- hyperparams  ---- #
            "input_dim": len(train_dataset.features),
            "hidden_dim1": hp.get("hidden_dim1", 128),
            "hidden_dim2": hp.get("hidden_dim2", 128),
            "num_layers": hp.get("num_layers", 1),
            "dropout": hp.get("dropout", 0.1),
            "batch_size": hp.get("batch_size",64),
            "seq_len": hp.get("seq_len",10),
            "lr": hp.get("lr", 1e-3),
            "epochs": hp.get("epochs", 20),
            "opt": hp.get("opt", "adam"),
            "device": device
            # -------------------- #
            }


        #Initiate MLFlow run
        run = start_mlflow_run(run_params)
        start_time = time.time()

        
        print(f"\n🔎 Training with model {model.__class__.__name__} params: {hp}")
        train_preds, train_true_rul, val_preds, val_true_rul, train_loss, val_loss = train_model(
            model, train_loader, val_loader,
            epochs=hp.get("epochs", 20),
            lr=hp.get("lr", 1e-3),
            opt=hp.get("opt", "adam"),
            device=device,
            mlflow=1
        )
        train_secs = time.time() - start_time
        mlflow.log_metric("train_time_sec", train_secs)
        mlflow.end_run()
        results.append((hp, train_loss))

        if val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            best_model = model
            best_params = hp

    
    print("\n✅ Grid Search Complete on model:", model_class.__name__)
    print(f"Best Params: {results[min(range(len(results)), key=lambda i: results[i][1])][0]}")
    print(f"Best Val Loss: {best_loss:.4f}\n")

    return best_model, best_params, best_loss, results

def plot_loss_curves(train_loss, val_loss, epochs, val_true_rul, val_preds):
    plt.figure(figsize=(14,6))
    # Training
    plt.subplot(1,2,1)
    plt.plot(np.arange(1, epochs+1), train_loss, label="Training loss")
    plt.plot(np.arange(1, epochs+1), val_loss, label="Validation loss")
    plt.title("MSE losses: Training vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.grid(visible=True)
    # plt.xticks(range(1,epochs+1))
    plt.legend()

    # Validation
    plt.subplot(1,2,2)
    plt.plot(val_true_rul[:600], label="Actual RUL")
    plt.plot(val_preds[:600], label="Predicted RUL")
    plt.title("Validation Set: Predicted vs Actual RUL")
    plt.xlabel("Sample")
    plt.ylabel("Remaining Useful Life (days)")
    plt.grid(visible=True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_test_results(true_rul, preds, model):
    plt.figure(figsize=(12,5))
    plt.plot(true_rul[:300], label="True RUL")
    plt.plot(preds[:300], label="Predicted RUL")
    plt.xlabel("Test Sample")
    plt.ylabel("RUL")
    plt.grid(visible=True)
    plt.title("Predicted vs True RUL on Test Set with " + str(model.__class__.__name__) + " model")
    plt.legend()
    plt.show()

def plot_bias_variance(train_loss, val_loss, epochs):
    # Bias and Variance
    plt.figure(figsize=(14,6))
    variance = [abs(val_loss[i] - train_loss[i]) for i in range(len(val_loss))]
    # Training
    plt.plot(np.arange(1, epochs+1), train_loss, label="Bias")
    plt.plot(np.arange(1, epochs+1), variance, label="Variance")
    plt.title("Bias vs Variance")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid(visible=True)
    # plt.xticks(range(1,epochs+1))
    plt.legend()
    plt.show()
