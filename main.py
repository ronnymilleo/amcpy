import pickle

import matplotlib.pyplot as plt
import numpy as np

import features as ft

# Config
frame_size = 1024
number_of_frames = 256
number_of_features = 9
modulations = ['8PSK', '16PSK', '16QAM', '64QAM', '256QAM', 'BPSK', 'QPSK']

for modulation_number in range(len(modulations)):
    pkl_file_name = 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\data\\' + \
                    modulations[modulation_number] + \
                    '_ALL_SNR.pickle'

    # Load the pickle file
    with open(pkl_file_name, 'rb') as handle:
        data = pickle.load(handle)

    # Quick code to separate frames
    signal = np.empty([len(data), number_of_frames, frame_size, 2])
    for a in range(len(data)):
        for b in range(number_of_frames):
            signal[a, b, :, :] = data[a][b, :, :]

    # Parse frames
    frame = np.zeros((len(data), number_of_frames, frame_size), dtype=np.complex)
    for snr in range(len(signal)):
        for i in range(number_of_frames):
            for j in range(frame_size):
                frame[snr, i, j] = (complex(signal[snr, i, j, 0], signal[snr, i, j, 1]))  # complex(Real, Imaginary)

    # Plot frame
    plt.plot(frame[5, 0, 0:200].real)
    plt.plot(frame[5, 0, 0:200].imag)
    plt.show()
    plt.plot(frame[15, 0, 0:200].real)
    plt.plot(frame[15, 0, 0:200].imag)
    plt.show()
    plt.plot(frame[25, 0, 0:200].real)
    plt.plot(frame[25, 0, 0:200].imag)
    plt.show()

    # Scatter plot
    plt.scatter(frame[25, 0, 0:200].real, frame[25, 0, 0:200].imag)
    plt.show()

    # Calculate features
    features = np.zeros((len(frame), number_of_frames, number_of_features))
    for i in range(len(features)):
        for j in range(number_of_frames):
            features[i, j, :] = ft.calculate_features(frame[i, j, :])

    # Save the samples ina pickle file
    with open('C:\\Users\\ronny\\PycharmProjects\\amcpy\\data\\' +
              modulations[modulation_number] +
              '_features.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
