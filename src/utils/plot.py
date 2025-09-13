import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple, Sequence, Union
from sklearn.cluster import KMeans
from models.similarity import ResidualSimilarityModel
from utils.process import txt_to_pd, normalized_sensor, save_fig, generate_gif
import pandas as pd


def plot_all(train):
    # Plot Remaining Life vs Sensor Measurement of all units and all clusters
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20)) 
    for i in range(len(train.columns)-5):
        # Scatter plot on the ith subplot on cluster 0
        axs[i//5, i-(i//5)*5].scatter(train['Remaining Life'], train[train.columns[i+4]], 
                    color='skyblue', alpha=0.7, edgecolors='black', linewidths=0.5)
        axs[i//5, i-(i//5)*5].set_xlabel('Time')
        axs[i//5, i-(i//5)*5].set_ylabel(train.columns[i+4])
    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    plt.grid(True)
    save_fig("plot_all", tight_layout=True, fig_extension="png", resolution=300)

def plot_clusters(op_data):
    # Create and fit K-Means model using the optimal cluster number (k=6)
    kmeans = KMeans(n_clusters=6, random_state=42)
    labels = kmeans.fit_predict(op_data)

    # Get cluster centers
    centers = kmeans.cluster_centers_

    # Plot k-clusters 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Create 3D scatter plot
    ax.scatter(op_data[op_data.columns[0]], op_data[op_data.columns[1]], op_data[op_data.columns[2]], c=labels, cmap='viridis', s=50)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.grid(True)
    save_fig("plot_clusters", tight_layout=True, fig_extension="png", resolution=300)


def plot_single_cluster(train, op_data):
    # Create and fit K-Means model using the optimal cluster number (k=6)
    kmeans = KMeans(n_clusters=6, random_state=42)
    labels = kmeans.fit_predict(op_data)

    train['cluster'] = labels
    # Plot Remaining Life vs Sensor Measurement of all units and cluster 0
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20)) 

    for i in range(len(train.columns)-5):
        # Scatter plot on the ith subplot on cluster 0
        axs[i//5, i-(i//5)*5].scatter(train[train['cluster'] == 0]['Remaining Life'], train[train['cluster'] == 0][train.columns[i+4]], 
                    color='skyblue', alpha=0.7, edgecolors='black', linewidths=0.5)
        axs[i//5, i-(i//5)*5].set_xlabel('Time')
        axs[i//5, i-(i//5)*5].set_ylabel(train.columns[i+4])
        axs[i//5, i-(i//5)*5].grid(True)
    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    save_fig("plot_single_cluster", tight_layout=True, fig_extension="png", resolution=300)


def plot_health_indicator(train_fused):
    # Plot
    plt.figure()
    for i in range(len(train_fused[:10])):
        plt.plot(np.arange(1, len(train_fused[i])+1), train_fused[i], label="Engine " + str(i+1)) 
        plt.plot(len(train_fused[i])+1, train_fused[i].iloc[-1], marker='X', color="black")        # actual data points
    plt.xlabel('Operating Cycles')
    plt.ylabel('Health Indicator')
    # plt.ylim(0)
    plt.legend()
    plt.grid(True)
    plt.title('Training data Health Indicator')
    save_fig("plot_health_indicator", tight_layout=True, fig_extension="png", resolution=300)


def plot_rul(model: ResidualSimilarityModel,
            df: pd.DataFrame,
            train_fused: pd.DataFrame,
            trueRUL: Optional[float],
            frac: Optional[float],
            bins: int = 20):
                        
    # Residual-similarity model
    estRUL, ciRUL, __, nn_idx, pdfRUL = model.predict_rul_distribution(df)

    fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=False)
    labels = 1
    # --- Left: Histogram ---
    for i in nn_idx:
        if labels == 1:
            axes[0].plot(np.arange(1, len(train_fused[i])+1), train_fused[i], alpha= 0.4, color="lightblue", label="Nearest Neighbor Engine")       # actual data points
            labels = 0
        else:
            axes[0].plot(np.arange(1, len(train_fused[i])+1), train_fused[i], alpha= 0.4, color="lightblue")       # actual data points
        axes[0].plot(len(train_fused[i])+1, train_fused[i].iloc[-1], marker='X', color="black")       # actual data points
    axes[0].plot(np.arange(1, len(df)+1), df, label="Validation Engine")       # actual data points
    axes[0].set_xlabel('Operating Cycles')
    axes[0].set_ylabel('Health Indicator')
    axes[0].set_ylim(0)
    axes[0].set_xlim(0, 350)
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title('Engine Health Indicator at ' + str(frac) + '% Life')

    # --- Right: RUL Distribution ---
    if isinstance(pdfRUL, tuple) and len(pdfRUL) == 2:
        xs, pdf = np.asarray(pdfRUL[0], dtype=float), np.asarray(pdfRUL[1], dtype=float)
        axes[1].plot(xs, pdf, linewidth=2, label="PDF")
        xmin, xmax = xs.min(), xs.max()
    else:
        samples = np.asarray(pdfRUL, dtype=float)
        axes[1].hist(samples, bins=bins, density=True, alpha=0.6, label="Neighbor RUL samples")
        xmin, xmax = samples.min(), samples.max()

    lo, hi = ciRUL
    for x, lbl, ls in [(estRUL, "Estimate", "-"),
                       (lo, "CI Low", "--"),
                       (hi, "CI High", "--")]:
        axes[1].axvline(x, linestyle=ls, linewidth=2, label=lbl)

    if trueRUL is not None:
        axes[1].axvline(trueRUL, linestyle=":", linewidth=2, label="True RUL")

    axes[1].set_xlim(max(0, xmin - 0.05*(xmax - xmin)), xmax + 0.05*(xmax - xmin))
    axes[0].set_ylim(0)
    axes[1].set_xlabel("RUL (cycles)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].set_title('Engine RUL Distribution from Neighbor Samples at ' + str(frac) + '% Life')
    axes[1].grid(True)

def plot_val(k, mdl, train_fused, val_fused, val_units):

    # Perform evaluation on validation set
    k=k
    mdl = ResidualSimilarityModel(k=k).fit(train_fused)
    # Evaluate on one validation engine at 50%, 70%, 90%
    vidx = val_units 

    prediction = pd.DataFrame({'5%': [], '10%': [], '15%': [], 
                            '20%': [], '25%': [], '30%': [], '35%': [], '40%': [], 
                            '45%': [], '50%': [], '55%': [], '60%': [], '65%': [], 
                            '70%': [], '75%': [], '80%': [], '85%': [], '90%': [], 
                            '95%': [], '100%': []})

    truth = pd.DataFrame({'5%': [], '10%': [], '15%': [], 
                            '20%': [], '25%': [], '30%': [], '35%': [], '40%': [], 
                            '45%': [], '50%': [], '55%': [], '60%': [], '65%': [], 
                            '70%': [], '75%': [], '80%': [], '85%': [], '90%': [], 
                            '95%': [], '100%': []})

    for vidx in range(len(val_fused)):
        series = val_fused[vidx]["health_indicator"].values
        n = len(series)
        ypred = []
        yhat = []
        # Plot
        for i in range(5,105,5):
            frac = i / 100
            L = int(math.ceil(n * frac))
            obs = pd.DataFrame({"health_indicator": series[:L]})
            est, (lo, hi), nn_ruls, nn_idx, neighbor_ruls = mdl.predict_rul_distribution(obs)
            true_rul = n - L
            ypred.append(est)
            yhat.append(true_rul)
        prediction.loc[len(prediction)] = ypred
        truth.loc[len(truth)] = yhat
    error = truth - prediction
    # Plot
    plt.figure(figsize=(14,10))
    plt.plot(error.mean().index, error.mean(), label="Mean Validation Error of " +str(len(val_fused)) + " Engines")      
    plt.fill_between(error.mean().index, error.min(), error.max(), alpha=0.3, label="Validation Error Band")
    plt.xlabel('Percent Operating Life')
    plt.ylabel('Error in cycles (True - Predicted)')
    # plt.ylim(0)
    plt.legend(fontsize=15)
    plt.title('True RUL vs Predicted RUL')
    plt.grid()
    save_fig("plot_val", tight_layout=True, fig_extension="png", resolution=300)
