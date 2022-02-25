import os
import pathlib
from os.path import join

import numpy as np

# Directory to keep modulations data
matlab_data_folder = pathlib.Path(join(os.getcwd(), 'mat-data'))
if not os.path.exists(matlab_data_folder):
    print(matlab_data_folder.__str__() + ' does not exist, thus it will be created')
    os.mkdir(matlab_data_folder)
# Use here the name of the file defined in the MATLAB script
matlab_data_filename = 'all_modulations.mat'

# Directory to keep the calculated features
calculated_features_folder = pathlib.Path(join(os.getcwd(), 'calculated-features'))
if not os.path.exists(calculated_features_folder):
    print(calculated_features_folder.__str__() + ' does not exist, thus it will be created')
    os.mkdir(calculated_features_folder)

# Directory to keep the data used in the ARM microcontroller
arm_data_folder = pathlib.Path(join(os.getcwd(), 'arm-data'))
if not os.path.exists(arm_data_folder):
    print(arm_data_folder.__str__() + ' does not exist, thus it will be created')
    os.mkdir(arm_data_folder)

# Directory to keep the generated neural networks
trained_ann_folder = pathlib.Path(join(os.getcwd(), 'ann'))
if not os.path.exists(trained_ann_folder):
    print(trained_ann_folder.__str__() + ' does not exist, thus it will be created')
    os.mkdir(trained_ann_folder)

# Directory to keep figures
figures_folder = pathlib.Path(join(os.getcwd(), 'figures'))
if not os.path.exists(figures_folder):
    print(figures_folder.__str__() + ' does not exist, thus it will be created')
    os.mkdir(figures_folder)

# Directory to keep features figures
feature_figures_folder = pathlib.Path(join(os.getcwd(), 'figures', 'features'))
if not os.path.exists(feature_figures_folder):
    print(feature_figures_folder.__str__() + ' does not exist, thus it will be created')
    os.mkdir(feature_figures_folder)

# Define the number of threads you want to use while calculating features
num_threads = 8

# This is the definition of the modulations used in the project as a list and a dictionary
# The list is used to plot, noise is not plotted
modulation_signals = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
modulation_signals_with_noise = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'WGN']

# Definition of the SNR dictionary
snr_values = {0: '-10',
              1: '-8',
              2: '-6',
              3: '-4',
              4: '-2',
              5: '0',
              6: '2',
              7: '4',
              8: '6',
              9: '8',
              10: '10',
              11: '12',
              12: '14',
              13: '16',
              14: '18',
              15: '20'}

# Shortcut to the number of SNR based on the SNR dict
number_of_snr = len(snr_values)

# Definition of the number of frames, this value must be the same number of frames generated by MATLAB
number_of_frames = 100

# Definition of the frame size, also needs to be the same as generated by MATLAB
frame_size = 2048

# Definition of the features using LaTeX
features_names = {
    1: "$\\gamma_{max}$",
    2: "$\\sigma_{ap}$",
    3: "$\\sigma_{dp}$",
    4: "$\\sigma_{aa}$",
    5: "$\\sigma_{af}$",
    6: "$X$",
    7: "$X_2$",
    8: "$\\mu_{42}^{a}$",
    9: "$\\mu_{42}^{f}$",
    10: "$C_{20}$",
    11: "$C_{21}$",
    12: "$C_{40}$",
    13: "$C_{41}$",
    14: "$C_{42}$",
    15: "$C_{60}$",
    16: "$C_{61}$",
    17: "$C_{62}$",
    18: "$C_{63}$"
}

# Collection of used features, since all features researched are implemented we want to select which will be used
used_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
used_features_names = []
for feature in used_features:
    used_features_names.append(features_names[feature])

# The feature's matrix will keep the calculated features data
features_matrix = np.zeros((number_of_snr, number_of_frames, len(used_features)), dtype=np.float32)

# Neural networks
features_files = [f + "_features" for f in modulation_signals_with_noise]

# The neural networks performs better if you train using higher SNR values
training_snr = np.linspace(10, 15, 6, dtype=int)
all_available_snr = np.linspace(0, 15, 16, dtype=int)
plotting_snr = all_available_snr

# Dictionary to access variables inside MAT file
mat_info = {'BPSK': 'signal_bpsk',
            'QPSK': 'signal_qpsk',
            '8PSK': 'signal_8psk',
            '16QAM': 'signal_qam16',
            '64QAM': 'signal_qam64',
            'WGN': 'signal_noise'}
