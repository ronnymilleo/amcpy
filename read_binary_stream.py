import json
import os
import pathlib
import pickle
from os.path import join

import numpy as np

with open("./info.json") as handle:
    infoJson = json.load(handle)

modulations = infoJson['modulations']['names']
snr = list(map(int, infoJson['snr']))
number_of_samples = 13109600  # Got from length of the dataset by SNR

# Convert binary files into pickle files
for modulation in modulations:
    signal = []
    for i, value in enumerate(snr):
        try:  # Look for file on default folder
            file_name = pathlib.Path(join(os.getcwd(),
                                          'gr-data',
                                          'binary',
                                          'binary_' + modulation + "(" + "{}".format(value) + ")"))
            data = np.fromfile(file_name, dtype=np.complex64)
        except FileNotFoundError:  # If exception is raised, then look for personal storage on Google Drive
            file_name = pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                          'gr-data',
                                          'binary',
                                          'binary_' + modulation + "(" + "{}".format(value) + ")"))
            continue
        try:
            # Complex64 because it's float32 on I and Q
            data = np.fromfile(file_name, dtype=np.complex64)
        except FileNotFoundError:  # If exception is raised, then skip file
            print('Tried to get: ' + '~\\binary_' + modulation + "(" + "{}".format(value) + ")")
            print('File not found!')
            continue

        # Starts from 300*8 to skip zero values at the beginning of GR dataset
        aux = np.zeros((len(snr), len(data[300 * 8:number_of_samples])), dtype=np.complex64)
        aux[i][:] = data[300 * 8:number_of_samples]
        signal.append(aux[i])
        print('~\\binary_' + modulation + ' appended...')

    # Save binary files in pickle (without delay)
    with open(pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                'gr-data',
                                'pickle',
                                modulation + '.pickle')), 'wb') as handle:
        pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(modulation + ' file saved...')
print('Finished.')
