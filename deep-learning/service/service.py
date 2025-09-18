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

# FastAPI app
app = FastAPI()

# ---------------------------
# Inference Endpoint
# ---------------------------
@app.post("/predict")
def predict(file: UploadFile = File(...)):

    """Handle TXT/CSV file upload"""
    print("Client connected")

    try:
        print("Try File read")

        contents =  file.file.read()
        decoded = contents.decode("utf-8")
        print("File read")
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
        print("data read")

        # Process data
        df = data_process(df)
        # Run prediction
        pred_dataset  = ServiceData(df, seq_len=seq_len)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        preds = []

        model.to(device)
        with torch.no_grad():
            for x_pred, __ in pred_loader:
                x_pred = x_pred.to(device)
                output = model(x_pred)
                preds.extend(output.cpu().numpy().ravel().tolist())
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Healthcheck Endpoint
# ---------------------------
# @app.get("/healthcheck")
# def healthcheck():
#     return {"status": "ok", "model_loaded": model is not None}

# # ---------------------------
# # Drift Detection (periodic use)
# # ---------------------------
# @app.post("/check-drift")
# def check_drift(new_data: list[SensorData]):
#     # Convert incoming batch to DataFrame
#     new_df = pd.DataFrame([d.values for d in new_data])
    
#     # Build Evidently drift report
#     report = Report([
#         DataDriftPreset(),
#         # you can add others
#     ])
    
#     my_eval = report.run(reference_data=ref_df, current_data=new_df)
#     drift_detected_flag = my_eval.as_dict()["metrics"][0]["result"]["dataset_drift"]
    
#     return {
#         "drift": bool(drift_detected_flag),
#         "details": my_eval.as_dict()["metrics"][0]["result"]
#     }

# # ---------------------------
# # Retrain Trigger (placeholder)
# # ---------------------------
# @app.post("/retrain")
# def retrain():
#     return {"status": "triggered", "message": "Retraining pipeline started."}

    