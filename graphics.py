import pickle
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from os.path import join
import os

# Config
frame_size = 1024
number_of_frames = 4096
number_of_features = 9
modulations = ['BPSK', 'QPSK', '16QAM']

selection = int(input('Press 1 to load features from PKL or any other number to load features from MAT: '))

for modulation_number in range(len(modulations)):
    # Filename setup
    if selection == 1:
        pkl_file_name = pathlib.Path(join(os.getcwd(), 'data', str(modulations[modulation_number]) + '_features_from_PKL.pickle'))
    else:
        pkl_file_name = pathlib.Path(join(os.getcwd(), 'data', str(modulations[modulation_number]) + '_features_from_MAT.pickle'))

    # Load the pickle file
    with open(pkl_file_name, 'rb') as handle:
        features = pickle.load(handle)

    # Calculate mean of features
    mean_features = np.ndarray([len(features), number_of_frames, number_of_features])
    min_features = np.ndarray([len(features), number_of_frames, number_of_features])
    max_features = np.ndarray([len(features), number_of_frames, number_of_features])
    for i in range(len(mean_features)):
        for j in range(number_of_features):
            mean_features[i, :, j] = np.mean(features[i, :, j])
            min_features[i, :, j] = np.min(features[i, :, j])
            max_features[i, :, j] = np.max(features[i, :, j])

    # SNR axis setup
    var = np.linspace(-20, 30, 26)
    snr_array = np.zeros([len(features), number_of_frames, number_of_features])
    for i in range(len(snr_array)):
        for j in range(number_of_frames):
            for k in range(number_of_features):
                snr_array[i, j, k] = var[i]

    # Plot graphics
    for n in range(number_of_features):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.plot(snr_array[:, n, n], mean_features[:, n, n], linewidth=1.0)
        # plt.grid(True)
        # plt.fill_between(snr_array[:, 0, 0],
        #                  min_features[:, 0, n],
        #                  max_features[:, 0, n],
        #                  alpha=0.2)
        # plt.errorbar(snr_array[:, 0, 0],
        #              mean_features[:, 0, n],
        #              yerr=[mean_features[:, 0, n] - min_features[:, 0, n],
        #                    max_features[:, 0, n] - mean_features[:, 0, n]],
        #              uplims=True,
        #              lolims=True)
        plt.xlabel('SNR')
        plt.ylabel('Value')
        plt.title('Feature ' + str(n + 1))
        plt.legend(modulations)

if selection == 1:
    # Save figures
    for n in range(number_of_features):
        plt.figure(num=n)
        #figure_name = pathlib.Path(join(os.getcwd(), 'data/figures/features_from_PKL', str(n) + '.png'))
        figure_name = 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\figures\\features_from_PKL' + \
                      str(n) + \
                      '.png'
        plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
else:
    # Save figures
    for n in range(number_of_features):
        plt.figure(num=n)
        #figure_name = pathlib.Path(join(os.getcwd(), 'data/figures/features_from_MAT', str(n) + '.png'))
        figure_name = 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\figures\\features_from_MAT' + \
                      str(n) + \
                      '.png'
        plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
