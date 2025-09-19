from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib   # or pickle
import numpy as np
import pandas as pd
from evidently import Report
from service.utils import ServiceData, data_process
from evidently.metrics import *        # or specific metrics
from evidently.presets import *        # e.g. DataDriftPreset
from models.rnn import RNNRUL
from models.lstm import LSTMRUL
from models.seq2seq import Seq2SeqRUL
from models.informer import InformerRUL, InformerEncoder
import io
import torch
from torch.utils.data import DataLoader, Dataset

# Load model
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)
seq_len = model.seq_len
batch_size = model.batch_size
device = model.device
print(f"Model loaded: {model}, seq_len: {seq_len}, batch_size: {batch_size}, device: {device}")
# FastAPI app
app = FastAPI()

# ---------------------------
# Inference Endpoint
# ---------------------------
@app.post("/predict")
def predict(file: UploadFile = File(...)):

    """Handle TXT/CSV file upload"""
    try:
        contents =  file.file.read()
        decoded = contents.decode("utf-8")
        print(io.StringIO(decoded))
        # Detect CSV or TXT
        if file.filename.endswith(".csv"):
            print("csv")
            df = pd.read_csv(io.StringIO(decoded))
        elif file.filename.endswith(".txt"):
            print("txt")
            cols = list(range(26))  # 26 columns
            df = pd.read_csv(io.StringIO(decoded), sep=r"\s+", header=None, names=cols) 
        else:
            raise HTTPException(status_code=400, detail="Only .csv or .txt files are supported")

        # Process data
        df = data_process(df)
        feature_cols = df.columns.difference(["unit", "cycle", "max_cycle", "RUL"])
        scaler = joblib.load("models/scaler.joblib")
        df[feature_cols]  = scaler.transform(df[feature_cols])
        # Run prediction
        pred_dataset  = ServiceData(df, seq_len=seq_len)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        preds = []
        true_rul = []
        model.to(device)
        with torch.no_grad():
            for x_pred, y_pred in pred_loader:
                # x_pred = x_pred.to(device)
                x_pred, y_pred = x_pred.to(device), y_pred.to(device)
                output = model(x_pred)
                preds.extend(output.cpu().numpy().ravel().tolist())
                true_rul.extend(y_pred.cpu().numpy().ravel().tolist())

        return {"predictions": preds, "true_rul": true_rul}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

