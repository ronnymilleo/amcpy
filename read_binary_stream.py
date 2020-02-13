import os
import pathlib
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

modulations = ['BPSK', 'QPSK', 'PSK8', 'QAM16']
initial_delay = 256
number_of_samples = 1024

# Plot to verify modulation signals
for modulation in modulations:
    file_name = pathlib.Path(join(os.getcwd(), 'gr-data', 'binary_' + modulation))
    # Complex64 because it's float32 on I and Q
    data = np.fromfile(file_name, dtype=np.complex64)
    plt.figure(num=1)
    plt.plot(np.real(data[initial_delay:number_of_samples + initial_delay]))
    plt.plot(np.imag(data[initial_delay:number_of_samples + initial_delay]))
    plt.show()
    plt.figure(num=2)
    plt.scatter(np.real(data[initial_delay:number_of_samples + initial_delay]),
                np.imag(data[initial_delay:number_of_samples + initial_delay]))
    plt.show()

    # Save binary files in pickle (without delay)
    with open(pathlib.Path(join(os.getcwd(), 'gr-data', modulation + '.pickle')), 'wb') as handle:
        pickle.dump(data[1:len(data)], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(modulation + ' file saved...')

print('Finished.')
