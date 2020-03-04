import os
import pathlib
import pickle
from os.path import join
import json
import matplotlib.pyplot as plt
import numpy as np


with open("./info.json") as handle:
    infoJson = json.load(handle)

modulations = infoJson['modulations']['names']
snr = list(map(int, infoJson['snr']))
initial_delay = 256
number_of_samples = 1000

# Plot to verify modulation signals
for modulation in modulations:
    signal = []
    for i, value in enumerate(snr):
        file_name = pathlib.Path(join(os.getcwd(), 'gr-data', "binary", 'binary_' + modulation + "(" + "{}".format(value) + ")"))
        # Complex64 because it's float32 on I and Q
        data = np.fromfile(file_name, dtype=np.complex64)

        aux = np.zeros((len(snr), len(data)), dtype=np.complex64)
        aux[i][:] = data[:]
        signal.append(aux[i])
        '''
        plt.figure(num=1)
        plt.plot(np.real(data[initial_delay:number_of_samples + initial_delay]))
        plt.plot(np.imag(data[initial_delay:number_of_samples + initial_delay]))
        plt.show()
        plt.figure(num=2)
        plt.scatter(np.real(data[initial_delay:number_of_samples + initial_delay]),
                    np.imag(data[initial_delay:number_of_samples + initial_delay]))
        plt.show()
        '''
        #Save binary files in pickle (without delay)
    with open(pathlib.Path(join(os.getcwd(), 'gr-data', "pickle", modulation + '.pickle')), 'wb') as handle:
        pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(modulation + ' file saved...')

print('Finished.')