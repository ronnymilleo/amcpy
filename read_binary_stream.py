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
number_of_samples = 8*1024*200

# Plot to verify modulation signals
for modulation in modulations:
    signal = []
    for i, value in enumerate(snr):
        file_name = pathlib.Path(
            join(os.getcwd(), 'gr-data', "binary", 'binary_' + modulation + "(" + "{}".format(value) + ")"))
        try:
            # Complex64 because it's float32 on I and Q
            data = np.fromfile(file_name, dtype=np.complex64)
        except FileNotFoundError:
            print('Tried to get: ' + '~\\binary_' + modulation + "(" + "{}".format(value) + ")")
            print('File not found!')
            continue

        aux = np.zeros((len(snr), len(data[0:number_of_samples])), dtype=np.complex64)
        aux[i][:] = data[0:number_of_samples]
        signal.append(aux[i])

    # Save binary files in pickle (without delay)
    with open(pathlib.Path(join(os.getcwd(), 'gr-data', "pickle", modulation + '.pickle')), 'wb') as handle:
        pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(modulation + ' file saved...')
print('Finished.')
