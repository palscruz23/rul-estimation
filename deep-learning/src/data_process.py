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

from utils.helper import load_phm08, PHM08RULDataset

train_df = load_phm08("../data/Challenge_Data/", "train.txt")
test_df = load_phm08("../data/Challenge_Data/", "test.txt")

# Extract RUL for training set
rul_per_unit = train_df.groupby(0)[1].max().reset_index()
rul_per_unit.columns = ["unit", "max_cycle"]
train_df = train_df.rename(columns={0: "unit", 1: "cycle"})
train_df = train_df.merge(rul_per_unit, on="unit", how="left")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]




