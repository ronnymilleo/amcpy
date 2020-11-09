import os
import pathlib
from os.path import join

import numpy as np

data_folder = pathlib.Path(join(os.getcwd(), "mat-data"))
arm_folder = pathlib.Path(join(os.getcwd(), "arm-data"))
rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
fig_folder = pathlib.Path(join(os.getcwd(), "figures"))

# Used to calculate features
num_threads = 6

signals = ['BPSK',
           'QPSK',
           'PSK8',
           'QAM16',
           'QAM64',
           'noise']

signals_labels = [0, 1, 2, 3, 4, 5]

# Dictionary to access variable inside MAT file
mat_info = {'BPSK': 'signal_bpsk',
            'QPSK': 'signal_qpsk',
            'PSK8': 'signal_8psk',
            'QAM16': 'signal_qam16',
            'QAM64': 'signal_qam64',
            'noise': 'signal_noise'}

SNR_values = {0: '-10',
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

training_SNR = np.int8(np.linspace(11, 15, 5))
testing_SNR = np.int8(np.linspace(0, 15, 16))
frame_size = 2048
number_of_testing_frames = 500
number_of_training_frames = 5000

# Load dataset from MATLAB
features_files = [f + "_features" for f in signals]
testing_features_files = [f + "_features_testing" for f in signals]

features_names = {
    0: "Std of the Instantaneous Phase",
    1: "Std of the Absolute Instantaneous Phase",
    2: "Std of the Instantaneous Frequency",
    3: "Std of the Absolute Instantaneous Frequency",
    4: "Std of the CN Instantaneous Amplitude",
    5: "Std of the Absolute CN Instantaneous Amplitude",
    6: "Mean Value of the Signal Magnitude",
    7: "Squared Mean of the Signal Magnitude",
    8: "Normalized Sqrt Value of Sum of Amplitude",
    9: "Ratio of I/Q Components",
    10: "Gmax",
    11: "Kurtosis of the Absolute Value",
    12: "Kurtosis of the Absolute Frequency",
    13: "Cumulant Order 20",
    14: "Cumulant Order 21",
    15: "Cumulant Order 40",
    16: "Cumulant Order 41",
    17: "Cumulant Order 42",
    18: "Cumulant Order 60",
    19: "Cumulant Order 61",
    20: "Cumulant Order 62",
    21: "Cumulant Order 63"
}

used_features = [
    3,
    5,
    6,
    11,
    15,
    17
]

features_functions = {
    0: "functions.std_dev_inst_phase(signal_input)",
    1: "functions.std_dev_abs_inst_phase(signal_input)",
    2: "functions.std_dev_inst_freq(signal_input)",
    3: "functions.std_dev_abs_inst_freq(signal_input)",
    4: "functions.std_dev_inst_cna(signal_input)",
    5: "functions.std_dev_abs_inst_cna(signal_input)",
    6: "functions.mean_of_signal_magnitude(signal_input)",
    7: "functions.squared_mean_of_signal_magnitude(signal_input)",
    8: "functions.normalized_sqrt_of_sum_of_amp(signal_input)",
    9: "functions.ratio_iq(signal_input)",
    10: "functions.gmax(signal_input)",
    11: "functions.kurtosis_of_abs_amplitude(signal_input)",
    12: "functions.kurtosis_of_abs_freq(signal_input)",
    13: "functions.cumulant_20(signal_input)",
    14: "functions.cumulant_21(signal_input)",
    15: "functions.cumulant_40(signal_input)",
    16: "functions.cumulant_41(signal_input)",
    17: "functions.cumulant_42(signal_input)",
    18: "functions.cumulant_60(signal_input)",
    19: "functions.cumulant_61(signal_input)",
    20: "functions.cumulant_62(signal_input)",
    21: "functions.cumulant_63(signal_input)"
}

number_of_used_features = len(used_features)
used_features_names = []
for feature in used_features:
    used_features_names.append(features_names[feature])

# Calculate range of fixed point numbers #QM.N
q_range = {}
resolution = []
for M in range(0, 7):
    N = 15 - M
    k = "Q{}.{}".format(M, N)
    q_range[k] = ([-2 ** (M - 1), 2 ** (M - 1) - 2 ** (-N)])
    resolution.append(2 ** (-N))
