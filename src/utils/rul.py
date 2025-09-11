import pandas as pd
import math
from sklearn.linear_model import LinearRegression
import numpy as np



def add_health_condition(df):
    rul_per_unit = df.groupby('Unit Number')['Time, in cycles'].max().reset_index()
    rul_per_unit.columns = ["Unit Number", "max_cycle"]
    x_sensor = df.merge(rul_per_unit, on="Unit Number", how="left")
    x_sensor["RUL"] = x_sensor["max_cycle"] - x_sensor["Time, in cycles"]
    x_sensor["Health Condition"] = x_sensor["RUL"] / x_sensor["max_cycle"]
    x_sensor = x_sensor.sort_values(by=['Unit Number', 'Time, in cycles'])
    y_sensor = x_sensor['RUL'].to_frame()
    x_sensor = x_sensor.drop(["Time, in cycles", "Remaining Life", "Cluster", "max_cycle"], axis=1)
    x_sensor = x_sensor.set_index("Unit Number")
    return x_sensor, y_sensor

def sensor_trendability(x_train):
    # Linear Regression on each sensor for RUL
    fusion_model = LinearRegression()
    slope = []

    for sensor in x_train.columns:
        if "Sensor Measurement" in sensor:
            sensor_slope = []
            for unit in x_train.index.unique():
                x = x_train[x_train.index == unit][sensor].to_frame()
                y = x_train[x_train.index == unit]['Health Condition'].to_frame()
                fusion_model.fit(x, y)
                fusion_model.coef_[0]
                sensor_slope.append(fusion_model.coef_[0])
            sensor_median = np.abs(float(np.median(sensor_slope)))
            slope.append((sensor_median, sensor))
            print(sensor, sensor_median)
    slope = sorted(slope, reverse=True)[:8]
    selected_sensors = [i[1] for i in slope]
    return selected_sensors

def sensor_fusion(x_train, df, selected_sensors):
    # Linear Regression for fused sensor
    model = LinearRegression()
    x = x_train[selected_sensors]
    y = x_train["Health Condition"]
    model.fit(x, y)

    fused_list = []
    for i in df.index.unique():
        out = df[df.index == i][selected_sensors].values @ model.coef_
        out = pd.Series(out).rolling(window=21, center=True, min_periods=1).mean().values
        out = out + 1.0 - out[0]
        fused = pd.DataFrame({"health_indicator": out})
        fused_list.append(fused)
    return fused_list






