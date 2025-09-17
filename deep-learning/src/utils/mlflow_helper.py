from typing import Dict, Any
import mlflow

def start_mlflow_run(params: Dict[str, Any]):
    """
    params: dict of hyperparameters and run context to record once per run.
    """
    run = mlflow.start_run()
    mlflow.set_tag("project", "RUL-estimation")
    mlflow.set_tag("author", "Pol John Cruz")
    mlflow.set_tag("dataset", "PHM08 Dataset")
    mlflow.log_params(params)
    return run

def log_epoch_metrics(epoch: int, train_loss: float, val_loss: float = None, extra: Dict[str, float] = None):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    if val_loss is not None:
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    if extra:
        for k, v in extra.items():
            mlflow.log_metric(k, v, step=epoch)
