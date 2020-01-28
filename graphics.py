import pickle

import matplotlib.pyplot as plt
import numpy as np

# Config
frame_size = 1024
number_of_frames = 256
number_of_features = 9
modulations = ['8PSK', '16PSK', '16QAM', '64QAM', '256QAM', 'BPSK', 'QPSK']

for modulation_number in range(len(modulations)):
    pkl_file_name = 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\data\\' + \
                    modulations[modulation_number] + \
                    '_features.pickle'

    # Load the pickle file
    with open(pkl_file_name, 'rb') as handle:
        features = pickle.load(handle)

    # Calculate mean of features
    mean_features = np.ndarray([len(features), number_of_frames, number_of_features])
    min_features = np.ndarray([len(features), number_of_frames, number_of_features])
    max_features = np.ndarray([len(features), number_of_frames, number_of_features])
    for i in range(len(mean_features)):
        for j in range(number_of_features):
            mean_features[i, :, j] = np.mean(features[i, :, j])
            min_features[i, :, j] = np.min(features[i, :, j])
            max_features[i, :, j] = np.max(features[i, :, j])

    # # Example
    # print('Desvio padrão do valor abs da componente não-linear da fase instantânea: \n' +
    #       str(mean_features[:, 0, 0]))
    # print('Desvio padrão do valor direto da componente não-linear da fase instantânea: \n' +
    #       str(mean_features[:, 0, 1]))
    # print('Desvio padrão do valor abs da componente não-linear da freq instantânea: \n' +
    #       str(mean_features[:, 0, 2]))
    # print('Desvio padrão do valor direto da componente não-linear da freq instantânea: \n' +
    #       str(mean_features[:, 0, 3]))
    # print('Curtose: \n' +
    #       str(mean_features[:, 0, 4]))
    # print('Valor máximo da DEP da amplitude instantânea normalizada e centralizada: \n' +
    #       str(mean_features[:, 0, 5]))
    # print('Média da amplitude instantânea normalizada centralizada ao quadrado: \n' +
    #       str(mean_features[:, 0, 6]))
    # print('Desvio padrão do valor abs da amplitude instantânea normalizada e centralizada: \n' +
    #       str(mean_features[:, 0, 7]))
    # print('Desvio padrão da amplitude instantânea normalizada e centralizada: \n' +
    #       str(mean_features[:, 0, 8]))

    var = np.linspace(-20, 30, 26)
    snr_array = np.zeros([len(features), number_of_frames, number_of_features])
    for i in range(len(snr_array)):
        for j in range(number_of_frames):
            for k in range(number_of_features):
                snr_array[i, j, k] = var[i]

    for n in range(number_of_features):
        plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
        if n == 5:
            plt.semilogy(snr_array[:, n, n], mean_features[:, n, n], linewidth=1.0)
        else:
            plt.plot(snr_array[:, n, n], mean_features[:, n, n], linewidth=1.0)
        plt.grid(True)
        # plt.fill_between(snr_array[:, 0, 0],
        #                  min_features[:, 0, n],
        #                  max_features[:, 0, n],
        #                  alpha=0.2)
        # plt.errorbar(snr_array[:, 0, 0],
        #              mean_features[:, 0, n],
        #              yerr=[mean_features[:, 0, n] - min_features[:, 0, n],
        #                    max_features[:, 0, n] - mean_features[:, 0, n]],
        #              uplims=True,
        #              lolims=True)
        plt.xlabel('SNR')
        plt.ylabel('Value')
        plt.title('Feature ' + str(n + 1))
        plt.legend(modulations)

for n in range(number_of_features):
    plt.figure(num=n)
    figure_name = 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\figures\\all_modulations_feature_' + \
                  str(n) + \
                  '.png'
    plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
