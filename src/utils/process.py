import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt



def txt_to_pd(file_path):
    """
    Load data from a .txt file and convert into a pandas DataFrame
    Args: 
        file_path (string): Path location of the file to be converted
    Returns:
        df (DataFrame): converted dataframe with index set to "Unit Number"
    """
    
    delimiter = " "

    df = pd.read_csv(file_path, 
               delimiter=delimiter, 
               header=None,
               encoding='utf-8')
    df = df.dropna(axis=1)
    df.columns = ["Unit Number", "Time, in cycles", "Operational Setting 1", "Operational Setting 2", "Operational Setting 3", "Sensor Measurement 1",
              "Sensor Measurement 2", "Sensor Measurement 3", "Sensor Measurement 4", "Sensor Measurement 5", "Sensor Measurement 6", "Sensor Measurement 7", 
              "Sensor Measurement 8", "Sensor Measurement 9", "Sensor Measurement 10", "Sensor Measurement 11", "Sensor Measurement 12", "Sensor Measurement 13", 
              "Sensor Measurement 14", "Sensor Measurement 15", "Sensor Measurement 16", "Sensor Measurement 17", "Sensor Measurement 18", "Sensor Measurement 19", 
              "Sensor Measurement 20", "Sensor Measurement 21"]
    df = df.set_index("Unit Number")
    return df

def normalized_sensor(df, unique_clusters, cluster_mean, cluster_std):
    # Get normalized sensor measurement
    scaled_temp= pd.DataFrame()
    scaled= pd.DataFrame()

    for i in range(len(unique_clusters)):
        scaled_temp = (df[df['cluster'] == i] - cluster_mean.loc['Cluster ' + str(i+1)]) / cluster_std.loc['Cluster ' + str(i+1)]
        scaled_temp['Time, in cycles Reference'] = df[df['cluster'] == i]['Time, in cycles']
        scaled_temp['Remaining Life Reference'] = df[df['cluster'] == i]['Remaining Life']
        scaled_temp['Cluster Reference'] = df[df['cluster'] == i]['cluster']
        scaled = pd.concat([scaled, scaled_temp], axis=0)
    scaled = scaled.drop(["Time, in cycles", "Operational Setting 1", "Operational Setting 2", "Operational Setting 3", 
                                    "cluster", "Cycle Max", "Remaining Life"], axis=1)
    scaled = scaled.fillna(0)
    scaled.rename(columns={'Remaining Life Reference': 'Remaining Life', 'Time, in cycles Reference': 'Time, in cycles', 
                                'Cluster Reference': 'Cluster'}, inplace=True)
    scaled = scaled.sort_values(by='Time, in cycles')
    return scaled

# Where to save the figures
PROJECT_ROOT_DIR = "."

IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "src/figures/")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=False, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def generate_gif():
    img_dir =  "src/figures/"  
    gif_path = "src/figures/RUL.gif" 
    img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    images = [Image.open(img_dir + img).convert('RGBA') for img in img_list]
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=1000, loop=0)
    print("RUL prediction has been completed.")