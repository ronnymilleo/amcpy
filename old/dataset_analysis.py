import os
import pathlib
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

# Config
frame_size = 1024
number_of_frames = 4096
number_of_features = 9
modulations = ['BPSK', 'QPSK', 'OQPSK', '8PSK', '16PSK', '32PSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']

for modulation_number in range(len(modulations)):
    pkl_file_name = pathlib.Path(join(os.getcwd(), "data", str(modulations[modulation_number]) + "_RAW.pickle"))

    # Load the pickle file
    with open((pkl_file_name), 'rb') as handle:
        data = pickle.load(handle)

    signal = np.empty([26, number_of_frames, frame_size, 2])

    start = 0
    end = 4096
    snr = 0
    while snr < len(signal):
        signal[snr, 0:4096, :, :] = data[0][start:end, :, :]
        start += 4096
        end += 4096
        snr += 1

    snr_array = np.linspace(-20, 30, 26)
    frame = 0
    for n in range(len(signal)):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        plt.title('SNR = {0} Frame = {1}'.format(snr_array[n], frame))
        plt.xlabel('Amostras no tempo')
        plt.ylabel('Amplitude do sinal')
        plt.legend(['Real', 'Imag'])
        plt.plot(signal[n, frame, 0:32, 0:2])
        figure_name = pathlib.Path(join(os.getcwd(), "figures", modulations[modulation_number], "timedomain" + str(int(snr_array[n]))))
        plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
        plt.close()
