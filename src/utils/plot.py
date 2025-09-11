import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Sequence, Union
from sklearn.cluster import KMeans
from models.similarity import ResidualSimilarityModel
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
    plt.show()

def plot_single_cluster(train):
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

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()

def plot_health_indicator(train_fused):
    # Plot
    for i in range(len(train_fused[:10])):
        plt.plot(np.arange(1, len(train_fused[i])+1), train_fused[i], label=i+1)       # actual data points
        plt.xlabel('X')
        plt.ylabel('y')
        # plt.ylim(0)
        plt.legend()
        plt.title('Training data Health Indicator')
        plt.show()

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