import streamlit as st
import time
from utils.process import txt_to_pd, normalized_sensor, save_fig, generate_gif


def main():
    st.set_page_config(page_title="Data Exploration and Processing", page_icon="📊")
    st.title("📊 Data Exploration and Processing")
    st.sidebar.markdown("[💻 GitHub Repository](https://github.com/palscruz23/rul-prediction)")

    # Dataset page
    st.header("PHM08 Prognostics Data Challenge Dataset")
    st.markdown(
        """
        - **218 engine units**: Each representing a unique turbofan engine.
        - **21 sensor measurements**: Including fan speed, pressure, temperature, and vibration parameters.
        - **Degradation trajectories**: Each engine's data spans from normal operation to failure.
        - **Training and test sets**: Provided for model development and evaluation.

        Source: [PHM08 Challenge Data Set](https://data.nasa.gov/dataset/phm-2008-challenge)"""
    )

    # Load training data
    train = txt_to_pd("data/Challenge_Data/train.txt")
    # Training data Dataframe
    st.markdown(
        """
        ### Raw date (first 10)
        """
    )
    st.dataframe(train.head(10))

    # Training data distribution
    st.markdown(
        """
        ### Training data distribution
        """
    )
    st.image("src/figures/train_hist.png", caption=None,  output_format ="png")

    # Data exploration
    st.header("Data Exploration")
    # Remaining Life vs Sensor Measurement of all units
    st.markdown(
        """
        ### Remaining Life vs Sensor Measurement of all units
        """
    )
    st.image("src/figures/plot_all.png", caption=None,  output_format ="png")

    # Clustering
    st.markdown(
        """
        ### Unsupervised learning using K-means Clustering
        Get Operating Clusters using unsupervised learning 
        """
    )
    st.image("src/figures/plot_clusters.png", caption=None,  output_format ="png")

    # Remaining Life vs Sensor Measurement of all units and cluster 0
    st.markdown(
        """
        ### Remaining Life vs Sensor Measurement of all units and cluster 0
        """
    )
    st.image("src/figures/plot_single_cluster.png", caption=None,  output_format ="png")


if __name__ == "__main__":
    main()