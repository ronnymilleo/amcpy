import os
import pathlib
from os.path import join

import numpy as np
import functions

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

training_SNR = np.int8(np.linspace(10, 15, 6))
# training_SNR = np.int8(np.linspace(0, 15, 16))
testing_SNR = np.int8(np.linspace(0, 15, 16))
frame_size = 2048
number_of_testing_frames = 1000
number_of_training_frames = 1000

# Load dataset from MATLAB
features_files = [f + "_features" for f in signals]
testing_features_files = [f + "_features" for f in signals]

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

not_used_features = [
    2,
    4,
    6,
    8,
    12,
    14,
]

used_features = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18
]

features_functions = {
    1: "functions.gmax(signal_input)",
    2: "functions.std_dev_abs_inst_phase(signal_input)",
    3: "functions.std_dev_inst_phase(signal_input)",
    4: "functions.std_dev_abs_inst_cna(signal_input)",
    5: "functions.std_dev_abs_inst_cnf(signal_input)",
    6: "functions.mean_of_signal_magnitude(signal_input)",
    7: "functions.normalized_sqrt_of_sum_of_amp(signal_input)",
    8: "functions.kurtosis_of_cn_amplitude(signal_input)",
    9: "functions.kurtosis_of_cn_freq(signal_input)",
    10: "functions.cumulant_20(signal_input)",
    11: "functions.cumulant_21(signal_input)",
    12: "functions.cumulant_40(signal_input)",
    13: "functions.cumulant_41(signal_input)",
    14: "functions.cumulant_42(signal_input)",
    15: "functions.cumulant_60(signal_input)",
    16: "functions.cumulant_61(signal_input)",
    17: "functions.cumulant_62(signal_input)",
    18: "functions.cumulant_63(signal_input)"
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
