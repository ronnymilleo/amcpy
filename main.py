import pickle

import matplotlib.pyplot as plt
import numpy as np

import features as ft

# Config
frame_size = 1024
number_of_frames = 100
number_of_features = 9

# Load the pickle file
with open('16QAM_SNR0dB.pickle', 'rb') as handle:
    data = pickle.load(handle)

# Quick code to separate frames
signal = []
a = 0
b = frame_size
for i in range(len(data) // frame_size):
    signal.append(data[a:b])
    a = a + frame_size
    b = b + frame_size

# Parse frames
frame = np.zeros((number_of_frames, frame_size), dtype=np.complex)
for k in range(number_of_frames):
    for i in range(frame_size):
        frame[k][i] = (complex(signal[k][i][0], signal[k][i][1]))  # complex(Real, Imaginary)

# Plot frame
plt.plot(frame[0][:].real)
plt.plot(frame[0][:].imag)
plt.show()

# Scatter plot
plt.scatter(frame[0][:].real, frame[0][:].imag)
plt.show()

# Calculate features
features = np.zeros((number_of_frames, number_of_features))
for i in range(number_of_frames):
    features[i][:] = ft.calculate_features(frame[i][:])

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

plt.subplot(3, 3, 1)
plt.plot(features[:, 0])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 2)
plt.plot(features[:, 1])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 3)
plt.plot(features[:, 2])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 4)
plt.plot(features[:, 3])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 5)
plt.plot(features[:, 4])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 6)
plt.plot(features[:, 5])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 7)
plt.plot(features[:, 6])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 8)
plt.plot(features[:, 7])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.subplot(3, 3, 9)
plt.plot(features[:, 8])
plt.xlabel('Frame')
plt.ylabel('Feature value')
plt.show()

# TODO: error bar
