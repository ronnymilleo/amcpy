import json
import os
import pathlib
import pickle
from os.path import join
from termcolor import colored

import numpy as np

with open("./info.json") as handle:
    info_json = json.load(handle)

modulations = info_json['modulations']['names']
snr = [info_json['snr']['values'][i] for i in info_json['snr']['using']]
number_of_samples = 13109600  # Got from length of the dataset by SNR
data_set = info_json['dataSetForTraining']

# Convert binary files into pickle files
for modulation in modulations:
    signal = []
    for i, value in enumerate(snr):
        try:  # Look for file on default folder
            if data_set == "rayleigh":
                file_name = pathlib.Path(join(os.getcwd(),
                                            'gr-data',
                                            'binary',
                                            'binary_' + modulation + "(" + "{}".format(value) + ")"))
                data = np.fromfile(file_name, dtype=np.complex64)
            if data_set == "awgn":
                file_name = pathlib.Path(join(os.getcwd(),
                                            'gr-data',
                                            'binary',
                                            'binary_awgn_' + modulation + "(" + "{}".format(value) + ")"))
                data = np.fromfile(file_name, dtype=np.complex64)
        except FileNotFoundError:  # If exception is raised, then look for personal storage on Google Drive
            if data_set == "rayleigh":
                file_name = pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                            'gr-data',
                                            'binary',
                                            'binary_' + modulation + "(" + "{}".format(value) + ")"))
            if data_set == "awgn":
                file_name = pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                            'gr-data',
                                            'binary',
                                            'binary_awgn_' + modulation + "(" + "{}".format(value) + ")"))
        try:
            # Complex64 because it's float32 on I and Q
            data = np.fromfile(file_name, dtype=np.complex64)
        except FileNotFoundError:  # If exception is raised, then skip file
            print('Tried to get: ' + modulation + "(" + "{}".format(value) + ")")
            print('File not found!')
            continue

        # Starts from 300*8 to skip zero values at the beginning of GR dataset
        aux = np.zeros((len(snr), len(data[300 * 8:number_of_samples])), dtype=np.complex64)
        aux[i][:] = data[300 * 8:number_of_samples]
        signal.append(aux[i])
        print('{} samples of ~\\'.format(len(data[300 * 8:number_of_samples]))
              + modulation + '({}) appended...'.format(value))

    # Save binary files in pickle (without delay)
    if data_set == "rayleigh":        
        with open(pathlib.Path(join(os.getcwd(),
                                    'gr-data',
                                    'pickle',
                                    modulation + '.pickle')), 'wb') as handle:
            pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(modulation + ' file saved...')
    if data_set == "awgn":        
        with open(pathlib.Path(join(os.getcwd(),
                                    'gr-data',
                                    'pickle',
                                    modulation + '_awgn.pickle')), 'wb') as handle:
            pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(modulation + ' file saved...')

if not data_set == info_json['dataSetForTesting']:
    print(colored('Warning:', 'yellow'), 
    "dasets for testing and training are different! You must run this code again changing the option in Training, letting it equal for Testing.")
print('Finished.')
