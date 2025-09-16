# Remaining Useful Life Estimation of Turbofan Engines

This repository contains a machine learning project aimed at predicting the Remaining Useful Life (RUL) of turbofan engines using the **PHM08 Prognostics Data Challenge Dataset** provided by NASA. Accurate RUL Estimation can enable proactive maintenance, reduce operational costs, and prevent unexpected failures.

## 📊 Dataset Overview

The PHM08 dataset includes:

- **218 engine units**: Each representing a unique turbofan engine.
- **21 sensor measurements**: Including fan speed, pressure, temperature, and vibration parameters.
- **Degradation trajectories**: Each engine's data spans from normal operation to failure.
- **Training and test sets**: Provided for model development and evaluation.

Source: [PHM08 Challenge Data Set](https://data.nasa.gov/dataset/phm-2008-challenge)

## 🚀 To get started with RUL predicton project:

1. Clone the repository:

   ```
   git clone estimate_rul
   cd rul-estimation
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   ```

3. Open streamlit app for the RUL Estimation visualization
    ```
   streamlit run src/Main.py
   ```

## 📚 Projects Overview
### <i>Remaining Useful Life Estimation using machine learning techniques</i>
 - Data Exploration
   - Load training and test data
   - Utilise unsupervised learning (K-means) to get operational parameter clusters
 - Data Processing
   - Split data set into training and validation sets
   - Apply Standard Scaler to normalise different sensor measurements
 - Remaining Useful Life Estimation
   - Construct Asset Health Indicator
   - Perform Linear Regression on each sensor to get trendability
   - Perform sensor fusion to develop health indicator
   - Develop Residual-similarity model using Degree-2 polynomial fit
   - Plot health indicator of validation engine from 5% to 100% operating life
- Notebook: 
   ```
   notebooks/RUL using ML.ipynb
   ```

#### 📉 RUL Estimation using ML Demo

 ![RUL Estimation Demo](src/figures/RUL/RUL.gif)

#### 📃 References
[1] [MATLAB Similarity-Based Remaining Useful Life Estimation](https://au.mathworks.com/help/predmaint/ug/similarity-based-remaining-useful-life-estimation.html)

[2] [A similarity-based prognostics approach for Remaining Useful Life estimation of engineered systems](https://ieeexplore.ieee.org/document/4711421)


 ### <i>Remaining Useful Life Estimation using deep learning techniques</i> (soon)
 <!-- - Data Processing
   - Load training and test data
   - Split data set into training and validation sets
   - Create PHM08RULDataset dataset class
 - Remaining Useful Life Estimation
   - Initiate ML flow experiment
   - Create model classes for RNN, LSTM, Seq2Seq and Informer
   - Prepare training and validation loops
   - Perform grid search for hyperparameter tuning
   - Select best model
   - Perform bias vs variance analysis
   - Perform Estimation on test data.
  - Notebook: 
      ```
      notebooks/RUL using DL.ipynb
      ``` -->

## 📜 License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

