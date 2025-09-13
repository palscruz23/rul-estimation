import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import math
import streamlit as st
import base64

from utils.rul import add_health_condition, sensor_trendability, sensor_fusion
from utils.process import txt_to_pd, normalized_sensor, save_fig, generate_gif
from utils.plot import plot_all, plot_clusters, plot_single_cluster, plot_health_indicator, plot_rul, plot_val
from models.similarity import ResidualSimilarityModel

def data_process(params, engine):

    # Load training and test data
    train = txt_to_pd("data/Challenge_Data/train.txt")
    test = txt_to_pd("data/Challenge_Data/test.txt")

    train.hist(bins=10, figsize=(20,15))
    save_fig("train_hist", tight_layout=True, fig_extension="png", resolution=300)

    # Data Exploration
    # Get Actual Remaining Life
    unique_list = train.index.unique().tolist()
    len(unique_list)
    for i in range(len(unique_list)):
        train.loc[i+1, 'Cycle Max'] = train.loc[i+1]['Time, in cycles'].max()
    train['Remaining Life'] = train['Time, in cycles'] - train['Cycle Max']
    plot_all(train)

    # Get operational data
    op_data = train[['Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3']]

    # Create and fit K-Means model using the optimal cluster number (k=6)
    plot_clusters(op_data)
    kmeans = KMeans(n_clusters=6, random_state=42)
    labels = kmeans.fit_predict(op_data)

    # Get cluster centers
    plot_single_cluster(train, op_data)
    centers = kmeans.cluster_centers_
    train['cluster'] = labels

    # Data Processing
    # Split data set by units
    units = train.index.unique()    # all engine IDs
    print(f"Total engines: {len(units)}")

    # First split into train+val vs test
    training_units, test_units = train_test_split(units, test_size=params[0], random_state=42)
    train_units, val_units = train_test_split(training_units, test_size=params[0], random_state=42)
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
    
    return x_train, x_val, x_test, val_units

def predict_rul(params, engine, x_train, x_val, x_test, val_units):

    # Perform Linear Regression on each sensor to get trendability
    # Select most trendable sensors
    selected_sensors = sensor_trendability(x_train, params[1])

    # Sensor fusion to develop health indicator
    train_fused = sensor_fusion(x_train, x_train, selected_sensors)
    val_fused = sensor_fusion(x_train, x_val, selected_sensors)
    test_fused = sensor_fusion(x_train, x_test, selected_sensors)
    plot_health_indicator(train_fused)

    # Residual-similarity model
    k=params[2]
    mdl = ResidualSimilarityModel(k=k).fit(train_fused)
    # Evaluate on one validation engine at 50%, 70%, 90%
    vidx = engine  # pick the 3rd one if available
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

    # Perform evaluation on validation set
    plot_val(k, mdl, train_fused, val_fused, val_units)

# from streamlit_gallery import apps, components
# from streamlit_gallery.utils.page import page_group

def main():
    st.set_page_config(page_title="RUL Prediction", page_icon="🚀",)
    st.title("🚀 RUL Prediction Home Page 🚀")
    st.sidebar.markdown("[💻 GitHub Repository](https://github.com/palscruz23/rul-prediction)")

    # Home page
    st.markdown(
        """
        <div style="text-align: justify;">
        Unplanned equipment failures do not just disrupt operations. These failures drive up costs, reduce productivity, and can even compromise safety. Equipment reliability and operational efficiency are crucial for optimising equipment productivity while controlling cost. Traditional approaches like reactive maintenance (fix it when it breaks) or preventive maintenance (service at fixed intervals) often fall short. They either lead to costly downtime or force maintenance teams to replace components that still have useful life left.
        <br><br>
        
        Predictive maintenance (PdM) provides a smarter alternative. By leveraging sensor data, machine learning, and advanced analytics, PdM enables organizations to forecast equipment health and anticipate failures before they occur. This approach ensures that maintenance is performed only when necessary, minimizing downtime while maximizing asset availability.
        <br>
        
        A key element of predictive maintenance is Remaining Useful Life (RUL) prediction. RUL models estimate how much longer a component or system can operate safely and effectively before it reaches the end of its life. 
        
        One powerful approach is the similarity-based RUL model. This approach uses operational parameters and sensor information from start of equipment life until it fails. Instead of relying purely on complex physics or massive amounts of labeled data, similarity models:

        - Compare the current condition of equipment with historical failure patterns.
        - Find the “closest match” in past data.
        - Estimate the RUL based on how those similar systems degraded over time.

        
        RUL prediction has the potential to create a strategic advantage — enabling higher reliability, safer operations, and better results.
        </div>
        """,
        unsafe_allow_html=True)
    

    # Initialization
    if 'run' not in st.session_state:
        st.session_state['run'] = 1
    
    if st.session_state['run'] == 1:
        params = [0.1, 8, 10]
        engine = 1
        if st.button("Start RUL Prediction"):
            with st.spinner("Processing data..."):
                x_train, x_val, x_test, val_units = data_process(params, engine)
                st.success("Data loaded and processed! Ready for predicting remaining life.")
            with st.spinner("Predicting RUL..."):
                predict_rul(params, engine, x_train, x_val, x_test, val_units)
                st.success("Remaining Useful Life prediction has been completed!")
                st.session_state['run'] = 0
    else:
        st.markdown(
            """
            Existing data and prediction is loaded.
            """

        )

if __name__ == "__main__":
    main()
