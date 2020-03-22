import json
import os
import pathlib
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

# Open json file with parameters
with open("./info.json") as handle:
    infoJson = json.load(handle)

# Config
frame_size = infoJson['frameSize']
number_of_modulations = len(infoJson['modulations']['index'])
number_of_snr = len(infoJson['snr'])
number_of_frames = infoJson['numberOfFrames']
number_of_features = len(infoJson['features']['using'])
modulation_names = infoJson['modulations']['names']
feature_names = infoJson['features']['names']

features = []
for modulation in range(number_of_modulations):
    # Filename setup
    pkl_file_name = pathlib.Path(join(os.getcwd(),
                                      'gr-data',
                                      'pickle',
                                      modulation_names[modulation] + '_features.pickle'))

    # Load the pickle file
    with open(pkl_file_name, 'rb') as handle:
        features.append(pickle.load(handle))

# Calculate mean, min and max of features by SNR and by modulation
mean_features = np.ndarray([number_of_modulations, number_of_snr, number_of_frames, number_of_features])
std_features = np.ndarray([number_of_modulations, number_of_snr, number_of_frames, number_of_features])
for m in range(number_of_modulations):
    for snr in range(number_of_snr):
        for ft in range(number_of_features):
            mean_features[m, snr, :, ft] = np.mean(features[m][snr, :, ft])
            std_features[m, snr, :, ft] = np.std(features[m][snr, :, ft])

# SNR axis setup
var = np.linspace(-20, 20, 21)
snr_array = np.ndarray([number_of_modulations, number_of_snr, number_of_frames, number_of_features])
for m in range(number_of_modulations):
    for snr in range(number_of_snr):
        for fr in range(number_of_frames):
            for ft in range(number_of_features):
                snr_array[m, snr, fr, ft] = var[snr]

# Plot graphics using only mean
for n in range(number_of_features):
    plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
    for m in range(number_of_modulations):
        plt.plot(snr_array[m, :, n, n], mean_features[m, :, n, n], linewidth=1.0)
    plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[n])
    plt.xlabel('SNR')
    plt.ylabel('Value')
    plt.legend(modulation_names)
    figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features', 'feature_' + str(n + 1) + '.png'))
    plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
    plt.close()

# Plot graphics with all frames
for n in range(number_of_features):
    plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
    for m in range(number_of_modulations):
        plt.plot(snr_array[m, :, :, n], mean_features[m, :, :, n], linewidth=1.0)
    plt.xlabel('SNR')
    plt.ylabel('Value')
    plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[n])
    # TODO: put modulation names in legend
    # plt.legend(modulation_names)
    figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features', 'feature_' + str(n + 1) + '_all_frames.png'))
    plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
    plt.close()

# Plot graphics with error bar using standard deviation
for n in range(number_of_features):
    plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
    for m in range(number_of_modulations):
        plt.errorbar(snr_array[m, :, n, n],
                     mean_features[m, :, n, n],
                     yerr=std_features[m, :, n, n])
    plt.xlabel('SNR')
    plt.ylabel('Value with sigma')
    plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[n])
    plt.legend(modulation_names)
    figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features', 'feature_' + str(n + 1) + '_error_bar.png'))
    plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
    plt.close()
