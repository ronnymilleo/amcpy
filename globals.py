import os
import pathlib
from os.path import join

import numpy as np

# Data folders
matlab_data_folder = pathlib.Path(join(os.getcwd(), 'mat-data'))
calculated_features_folder = pathlib.Path(join(os.getcwd(), 'calculated-features'))
arm_data_folder = pathlib.Path(join(os.getcwd(), 'arm-data'))
trained_ann_folder = pathlib.Path(join(os.getcwd(), 'ann'))
figures_folder = pathlib.Path(join(os.getcwd(), 'figures'))

# Globals
num_threads = 6
signals = {0: 'BPSK',
           1: 'QPSK',
           2: '8PSK',
           3: '16QAM',
           4: '64QAM',
           5: 'wgn'}
modulation_signals = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
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
number_of_snr = len(snr_values)
number_of_frames = 1000
frame_size = 2048
used_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
features_matrix = np.zeros((number_of_snr, number_of_frames, len(used_features)), dtype=np.float32)

# Neural networks
features_files = [f + "_best_features" for f in signals.values()]
training_snr = np.int8(np.linspace(10, 15, 6))
testing_snr = np.int8(np.linspace(0, 15, 16))

# Dictionary to access variable inside MAT file
mat_info = {'BPSK': 'signal_bpsk',
            'QPSK': 'signal_qpsk',
            '8PSK': 'signal_8psk',
            '16QAM': 'signal_qam16',
            '64QAM': 'signal_qam64',
            'wgn': 'signal_noise'}

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
