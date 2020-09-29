import json
import os
import pathlib
import pickle
from os.path import join

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loads JSON file with execution setup
with open("./info.json") as json_handle:
    info_json = json.load(json_handle)

# Config variables based on the JSON file
number_of_frames = info_json['numberOfFrames']
number_of_features = len(info_json['features']['using'])
number_of_snr = len(info_json['snr']['using'])
snr_list = info_json['snr']['using']
modulation_list = info_json['modulations']['names']
# Maybe it's useful
# snr_dict = {}
# for a in range(0, 21):
#     snr_dict[a] = info_json['snr']['values'][a]

# Load dataset from MATLAB
features_files = [f + "_features.pickle" for f in modulation_list]


def preprocess_data():  # Prepare the data for the magic
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))
    number_of_samples = number_of_frames * number_of_snr
    X = np.zeros((number_of_samples * len(modulation_list), number_of_features), dtype=np.float32)
    y = np.ndarray((number_of_samples * len(modulation_list),), dtype=np.int8)

    # Here each modulation file is loaded and all
    # frames to all SNR values are vertically stacked
    for i, mod in enumerate(features_files):
        print("Processing {} data".format(mod.split("_")[0]))  # Separate the word 'features' from modulation file
        with open(join(data_folder, mod), 'rb') as ft_handle:
            data = pickle.load(ft_handle)

        # Location of each modulation on input matrix based on their number of samples
        location = i * number_of_samples

        for snr in snr_list:
            for frame in range(number_of_frames):
                X[location, :] = np.float32(data[snr - (21 - number_of_snr)][frame][:])  # [SNR][frames][ft]
                location += 1

            # An array containing the encoded labels for each modulation
            start = i * number_of_samples
            end = start + number_of_samples
            for index in range(start, end):
                y[index] = i

    # Finally, the data is split into train and test samples and standardised for a better learning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("\nData shape:")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Instantiate StandardScaler
    scaler = StandardScaler()
    # Fit into data used for training, results are means and variances used to standardise the data
    scaler.fit(X_train)
    # Remove mean and variance from data_train
    standardised_data_train = scaler.transform(X_train)
    # Remove mean and variance from data_test using the same values (based on theoretical background)
    standardised_data_test = scaler.transform(X_test)

    return standardised_data_train, standardised_data_test, y_train, y_test, scaler
