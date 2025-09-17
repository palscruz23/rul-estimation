





model_name = "Seq2Seq"   # <-- change to "Informer", "RNN", "LSTM", or "Seq2Seq"
seq_len = 100
batch_size = 32
lr = 1e-2
epochs = 10
opt = "adam" # <-- change to "adam", "rms"
device="cuda" if torch.cuda.is_available() else "cpu"

num_features=len(train_dataset.features)

print(f"Using device: {device}")

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

# Sequencing
units = train_df["unit"].unique()   # all engine IDs
training_units, test_units = train_test_split(units, test_size=0.1, random_state=42)
train_units, val_units = train_test_split(training_units, test_size=0.1, random_state=42)

# Apply Feature scaling to transform the scaling of different features into similar scales.
feature_cols = train_df.columns.difference(["unit", "cycle", "max_cycle", "RUL"])
training_df = train_df[train_df["unit"].isin(train_units)].copy()
val_df   = train_df[train_df["unit"].isin(val_units)].copy()
test_df   = train_df[train_df["unit"].isin(test_units)].copy()

scaler = StandardScaler()
training_df[feature_cols] = scaler.fit_transform(training_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

train_dataset = PHM08RULDataset(training_df, seq_len=seq_len, unit_ids=train_units)
val_dataset   = PHM08RULDataset(val_df, seq_len=seq_len, unit_ids=val_units)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

print(f"Total engines: {len(units)}")
print(f"Total train engines: {len(train_units)}")
print(f"Total validation engines: {len(val_units)}")
print(f"Total test engines: {len(test_units)}")

print("Starting training...")
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
print("Training completed.")

import joblib
# Save model
joblib.dump(model, "model/model.pkl")
