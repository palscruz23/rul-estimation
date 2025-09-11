import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import math

from utils.rul import add_health_condition, sensor_trendability, sensor_fusion
from utils.process import txt_to_pd, normalized_sensor, save_fig, generate_gif
from utils.plot import plot_all, plot_clusters, plot_single_cluster, plot_health_indicator, plot_rul
from models.similarity import ResidualSimilarityModel



def main():

    # Load training and test data
    train = txt_to_pd("data/Challenge_Data/train.txt")
    test = txt_to_pd("data/Challenge_Data/test.txt")

    # Data Exploration
    # Get Actual Remaining Life
    unique_list = train.index.unique().tolist()
    len(unique_list)
    for i in range(len(unique_list)):
        train.loc[i+1, 'Cycle Max'] = train.loc[i+1]['Time, in cycles'].max()
    train['Remaining Life'] = train['Time, in cycles'] - train['Cycle Max']

    # Get operational data
    op_data = train[['Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3']]

    # Create and fit K-Means model using the optimal cluster number (k=6)
    kmeans = KMeans(n_clusters=6, random_state=42)
    labels = kmeans.fit_predict(op_data)

    # Get cluster centers
    centers = kmeans.cluster_centers_
    train['cluster'] = labels

    # Data Processing
    # Split data set by units
    units = train.index.unique()    # all engine IDs
    print(f"Total engines: {len(units)}")

    # First split into train+val vs test
    training_units, test_units = train_test_split(units, test_size=0.1, random_state=42)
    train_units, val_units = train_test_split(training_units, test_size=0.1, random_state=42)
    print(f"Total train engines: {len(train_units)}")
    print(f"Total val engines: {len(val_units)}")
    print(f"Total test engines: {len(test_units)}")

    # Split data set into training and validation sets
    # Apply random seed to allow repetition of output
    train_set = train[train.index.isin(train_units)].copy()
    val_set   = train[train.index.isin(val_units)].copy()
    test_set   = train[train.index.isin(test_units)].copy()

    # Apply Standard Scaler by cluster to transform the scaling of different features into similar scales.
    unique_clusters = train_set['cluster'].unique().tolist()

    cluster_mean= pd.DataFrame()
    cluster_std= pd.DataFrame()

    for i in range(len(unique_clusters)):
        cluster_mean['Cluster ' + str(i+1)] = train_set[train_set['cluster'] == i].mean()
        cluster_std['Cluster ' + str(i+1)] = train_set[train_set['cluster'] == i].std()

    cluster_mean = cluster_mean.transpose()
    cluster_std = cluster_std.transpose()

    # Get normalized sensor measurement
    train_scaled = normalized_sensor(train_set, unique_clusters, cluster_mean, cluster_std)
    val_scaled = normalized_sensor(val_set, unique_clusters, cluster_mean, cluster_std)
    test_scaled = normalized_sensor(test_set, unique_clusters, cluster_mean, cluster_std)

    # Convert training and test dataset into x and y
    xtrain = train_scaled.drop(["Remaining Life"], axis=1)
    ytrain = train_scaled["Remaining Life"]

    # Construct Asset Health Indicator
    x_train, y_train = add_health_condition(train_scaled)
    x_val, y_val = add_health_condition(val_scaled)
    x_test, y_test = add_health_condition(test_scaled)

    # Perform Linear Regression on each sensor to get trendability
    # Select most trendable sensors
    selected_sensors = sensor_trendability(x_train)

    # Sensor fusion to develop health indicator
    train_fused = sensor_fusion(x_train, x_train, selected_sensors)
    val_fused = sensor_fusion(x_train, x_val, selected_sensors)
    test_fused = sensor_fusion(x_train, x_test, selected_sensors)

    # plot_health_indicator(train_fused)

    # Residual-similarity model
    k=50
    mdl = ResidualSimilarityModel(k=k).fit(train_fused)
    # Evaluate on one validation engine at 50%, 70%, 90%
    vidx = 3 if len(val_fused) >= 3 else 0  # pick the 3rd one if available
    series = val_fused[vidx]["health_indicator"].values
    n = len(series)

    # Plot
    for i in range(5,105,5):
        frac = i / 100
        L = int(math.ceil(n * frac))
        obs = pd.DataFrame({"health_indicator": series[:L]})
        est, (lo, hi), nn_ruls, nn_idx, neighbor_ruls = mdl.predict_rul_distribution(obs)
        true_rul = n - L
        if i % 10 == 0:
            print(f"\nObserved {int(frac*100)}% of life:")
            print(f"  True RUL = {true_rul} cycles")
            print(f"  Estimated RUL (median of {len(nn_ruls)} neighbors) = {est:.1f}")
            print(f"  90% CI ≈ [{lo:.1f}, {hi:.1f}]")
        plot_rul(model=mdl, df=obs, train_fused=train_fused, trueRUL=true_rul, frac=int(frac*100), bins=15)
        save_fig("RUL " + str(frac), tight_layout=True, fig_extension="png", resolution=300)

    # Generate gif from 5-100% life
    generate_gif()


if __name__ == "__main__":
    main()

