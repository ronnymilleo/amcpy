import json
import os
import pathlib
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

# Open json file with parameters
with open("./info.json") as handle:
    info_json = json.load(handle)

# Config
data_set = info_json['dataSetForTraining']
modulations = info_json['modulations']['names']
snr_list = info_json['snr']['using']
frame_size = info_json['frameSize']
number_of_frames = info_json['numberOfFrames']
feature_names = info_json['features']['names']
number_of_features = len(info_json['features']['using'])


def load_files():
    loaded_data = []
    for mod in modulations:
        file_name = pathlib.Path(join(os.getcwd(), 'mat-data', 'pickle', mod + '_features.pickle'))
        with open(file_name, 'rb') as file:
            loaded_data.append(pickle.load(file))
        print(mod + ' file loaded...')
    return loaded_data


def calculate_features_mean(data):
    ft_array = np.array(data)
    ft_mean_array = np.ndarray((len(modulations), len(snr_list), 1, number_of_features))
    for i in range(len(modulations)):
        for j in range(len(snr_list)):
            for k in range(number_of_features):
                ft_mean_array[i, j, 0, k] = np.mean(ft_array[i, j, :, k])
    return ft_mean_array


def calculate_features_stddev(data):
    ft_array = np.array(data)
    ft_mean_array = np.ndarray((len(modulations), len(snr_list), 1, number_of_features))
    for i in range(len(modulations)):
        for j in range(len(snr_list)):
            for k in range(number_of_features):
                ft_mean_array[i, j, 0, k] = np.std(ft_array[i, j, :, k])
    return ft_mean_array


def generate_snr_axis():
    snr_values = np.linspace((snr_list[0] - 10) * 2, (snr_list[-1] - 10) * 2, len(snr_list))
    # Repeat x_axis for all modulations in data
    x_axis = np.ndarray((len(modulations), len(snr_list)))
    for i in range(len(modulations)):
        x_axis[i, :] = snr_values
    return x_axis


def simple_plot(snr_axis, data_axis):
    # Plot graphics using only mean
    for n in range(number_of_features):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.plot(snr_axis[0, :], data_axis[0, :, 0, n], '#03cffc', linewidth=1.0, antialiased=True)  # BPSK
        plt.plot(snr_axis[1, :], data_axis[1, :, 0, n], '#6203fc', linewidth=1.0, antialiased=True)  # QPSK
        plt.plot(snr_axis[2, :], data_axis[2, :, 0, n], '#be03fc', linewidth=1.0, antialiased=True)  # PSK8
        plt.plot(snr_axis[3, :], data_axis[3, :, 0, n], '#fc0320', linewidth=1.0, antialiased=True)  # QAM16
        plt.plot(snr_axis[4, :], data_axis[4, :, 0, n], 'g', linewidth=1.0, antialiased=True)  # QAM64
        plt.plot(snr_axis[5, :], data_axis[5, :, 0, n], 'k', linewidth=1.0, antialiased=True)  # Noise
        plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[n])
        plt.xlabel('SNR')
        plt.ylabel('Value')
        plt.legend(modulations)
        figure_name = pathlib.Path(join(os.getcwd(),
                                        'figures',
                                        'features',
                                        'ft_{}_SNR_({})_a_({})_mean.png'.format(str(n + 1),
                                                                                (snr_list[0] - 10) * 2,
                                                                                (snr_list[-1] - 10) * 2)))
        plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                    orientation='landscape', format='png',
                    transparent=False, bbox_inches=None, pad_inches=0.1)
        plt.close()
        print('Plotting means of feature number {}'.format(n))


def n_frames_plot(n_frames, snr_axis, data_axis):
    for n in range(number_of_features):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.plot(snr_axis[0, :], data_axis[0, :, 0:n_frames, n], '#03cffc', linewidth=1.0, antialiased=True)  # BPSK
        plt.plot(snr_axis[1, :], data_axis[1, :, 0:n_frames, n], '#6203fc', linewidth=1.0, antialiased=True)  # QPSK
        plt.plot(snr_axis[2, :], data_axis[2, :, 0:n_frames, n], '#be03fc', linewidth=1.0, antialiased=True)  # PSK8
        plt.plot(snr_axis[3, :], data_axis[3, :, 0:n_frames, n], '#fc0320', linewidth=1.0, antialiased=True)  # QAM16
        plt.plot(snr_axis[4, :], data_axis[4, :, 0:n_frames, n], 'g', linewidth=1.0, antialiased=True)  # QAM64
        plt.plot(snr_axis[5, :], data_axis[5, :, 0:n_frames, n], 'k', linewidth=1.0, antialiased=True)  # Noise
        plt.xlabel('SNR')
        plt.ylabel('Value')
        plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[n])
        # TODO: put modulation names in legend
        # plt.legend(modulation_names)

        figure_name = pathlib.Path(join(os.getcwd(),
                                        'figures',
                                        'features',
                                        'ft_{}_SNR_({})_a_({})_{}_frames.png'.format(str(n + 1),
                                                                                     (snr_list[0] - 10) * 2,
                                                                                     (snr_list[-1] - 10) * 2,
                                                                                     n_frames)))
        plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                    orientation='landscape', format='png',
                    transparent=False, bbox_inches=None, pad_inches=0.1)
        plt.close()
        print('Plotting 500 frames of feature number {}'.format(n))


def errorbar_plot(snr_axis, mean, stddev):
    # Plot graphics with error bar using standard deviation
    for n in range(number_of_features):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.errorbar(snr_axis[0, :],
                     mean[0, :, 0, n],
                     yerr=stddev[0, :, 0, n], color='#03cffc')
        plt.errorbar(snr_axis[1, :],
                     mean[1, :, 0, n],
                     yerr=stddev[1, :, 0, n], color='#6203fc')
        plt.errorbar(snr_axis[2, :],
                     mean[2, :, 0, n],
                     yerr=stddev[2, :, 0, n], color='#be03fc')
        plt.errorbar(snr_axis[3, :],
                     mean[3, :, 0, n],
                     yerr=stddev[3, :, 0, n], color='#fc0320')
        plt.errorbar(snr_axis[4, :],
                     mean[4, :, 0, n],
                     yerr=stddev[4, :, 0, n], color='g')
        plt.errorbar(snr_axis[5, :],
                     mean[5, :, 0, n],
                     yerr=stddev[5, :, 0, n], color='k')
        plt.xlabel('SNR')
        plt.ylabel('Value with sigma')
        plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[n])
        plt.legend(modulations)
        figure_name = pathlib.Path(join(os.getcwd(),
                                        'figures',
                                        'features',
                                        'ft_{}_SNR_({})_a_({})_err.png'.format(str(n + 1),
                                                                               (snr_list[0] - 10) * 2,
                                                                               (snr_list[-1] - 10) * 2)))
        plt.savefig(figure_name, dpi=300, facecolor='w', edgecolor='w',
                    orientation='landscape', format='png',
                    transparent=False, bbox_inches=None, pad_inches=0.1)
        plt.close()
        print('Plotting error bar of feature number {}'.format(n))


if __name__ == '__main__':
    # Load files
    files = load_files()
    # Process
    snr_array = generate_snr_axis()
    mean_array = calculate_features_mean(files)
    std_array = calculate_features_stddev(files)
    # Plot
    simple_plot(snr_array, mean_array)
    n_frames_plot(100, snr_array, np.array(files))
    errorbar_plot(snr_array, mean_array, std_array)
