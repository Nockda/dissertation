import matplotlib.pyplot as plt
from IPython import display
import gym

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import csv
from tqdm import tqdm

import tensorflow as tf
from tensorflow import shape,math
from tensorflow.keras import Input,layers,Model
from tensorflow.keras.losses import mse,binary_crossentropy
from tensorflow.keras.utils import plot_model

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import os
import pandas as pd
import numpy as np

output_dir = os.path.join(".", "output_front")  # Path to the output directory
subdirs = [f.path for f in os.scandir(output_dir) if f.is_dir()]
subdirs.sort()

# Create an empty 3D array to store the combined data
combined_arr = np.empty((len(subdirs), 1000, 10))

# Loop through each subdirectory and load the CSV files
for i, subdir in enumerate(subdirs):
    action_filename = os.path.join(subdir, "action.csv")
    obs_filename = os.path.join(subdir, "obs.csv")

    # Load the action and obs CSV files
    action_df = pd.read_csv(action_filename,  header=None)
    obs_df = pd.read_csv(obs_filename,  header=None)

    # Concatenate the DataFrames horizontally
    combined_data = pd.concat([action_df, obs_df], axis=1)

    # Convert combined_data to a 3D array and assign it to combined_arr
    combined_arr[i-1] = np.reshape(combined_data.values, (1000, 10))

# Print the shape of combined_arr
print(combined_arr.shape)
flattened_arr = combined_arr.reshape(10000, 10000)

combined_df = np.array(combined_arr)

import torch

# Convert combined_arr to PyTorch Tensor
combined_tensor = torch.from_numpy(combined_arr)

# Print the shape of combined_tensor
print(combined_tensor.shape)

import matplotlib.pyplot as plt
from IPython import display
import gym

action_sp = combined_data.iloc[:, :2]
obs_sp = combined_data.iloc[:, 2:]

env = gym.make('Swimmer-v3', render_mode = 'human')

# Iterate through the rows
for i in range(len(action_sp)):
    # Get the i-th row
    action = action_sp.iloc[i]
    observation = obs_sp.iloc[i]
    print(action)

    # If this is the first iteration, set the environment state to the given observation
    # Note: This assumes that the observation you've stored is the entire state that can be set with `env.reset()`
    # If this is not the case, you cannot simply set the environment state to the observation
    if i == 0:
        env.reset()  # We ignore the initial observation returned by `reset`

    # Apply the action
    next_observation, reward, done, trunc, info = env.step(action)
    # Render the environment
    env.render()
    # If you want to slow down each step for viewing, you can use time.sleep
    # time.sleep(0.01)

# Close the environment
env.close()
