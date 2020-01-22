import pickle

import matplotlib.pyplot as plt
import numpy as np

import features as ft

# Config
frame_size = 1024
number_of_frames = 100
number_of_features = 9

# Load the pickle file
with open('C:\\Users\\ronny\\PycharmProjects\\amcpy\\data\\16QAM_ALL_SNR.pickle', 'rb') as handle:
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

# Calculate mean of features
mean_features = np.ndarray([len(frame), number_of_frames, number_of_features])
std_features = np.ndarray([len(frame), number_of_frames, number_of_features])
min_features = np.ndarray([len(frame), number_of_frames, number_of_features])
max_features = np.ndarray([len(frame), number_of_frames, number_of_features])
for i in range(len(mean_features)):
    for j in range(number_of_features):
        mean_features[i, :, j] = np.mean(features[i, :, j])
        std_features[i, :, j] = np.std(features[i, :, j])
        min_features[i, :, j] = np.min(features[i, :, j])
        max_features[i, :, j] = np.max(features[i, :, j])

# Example
print('Desvio padrão do valor absoluto da componente não-linear da fase instantânea: ' + str(features[0, 0]))
print('Desvio padrão do valor direto da componente não-linear da fase instantânea: ' + str(features[0, 1]))
print('Desvio padrão do valor absoluto da componente não-linear da frequência instantânea: ' + str(features[0, 2]))
print('Desvio padrão do valor direto da componente não-linear da frequência instantânea: ' + str(features[0, 3]))
print('Curtose: ' + str(features[0, 4]))
print('Valor máximo da DEP da amplitude instantânea normalizada e centralizada: ' + str(features[0, 5]))
print('Média da amplitude instantânea normalizada centralizada ao quadrado: ' + str(features[0, 6]))
print('Desvio padrão do valor absoluto da amplitude instantânea normalizada e centralizada: ' + str(features[0, 7]))
print('Desvio padrão da amplitude instantânea normalizada e centralizada: ' + str(features[0, 8]))

var = np.linspace(-20, 30, 26)
snr_array = np.zeros([len(frame), number_of_frames, number_of_features])
for i in range(len(snr_array)):
    for j in range(number_of_frames):
        for k in range(number_of_features):
            snr_array[i, j, k] = var[i]

for n in range(number_of_features):
    plt.subplot(1, 2, 1)
    # plt.plot(snr_array[:, :, n], features[:, :, n])
    plt.fill_between(snr_array[:, 0, 0],
                     min_features[:, 0, n],
                     max_features[:, 0, n],
                     color='gray',
                     alpha=0.2)
    plt.xlabel('SNR')
    plt.ylabel('Value')
    plt.subplot(1, 2, 2)
    # plt.plot(snr_array[:, :, n], features[:, :, n])
    plt.errorbar(snr_array[:, 0, 0], mean_features[:, 0, n], yerr=3 * std_features[:, 0, n], uplims=True, lolims=True)
    plt.xlabel('SNR')
    plt.ylabel('Value')
    plt.show()
