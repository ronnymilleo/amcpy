import pickle
import numpy as np
import scipy.io
import pathlib
from os.path import join
import features as ft

# Config
frame_size = 1024
number_of_frames = 64
number_of_snr = 26
number_of_features = 22
modulations = ['BPSK', 'QPSK', '16QAM']

selection = int(input('Press 1 to load raw PKL or any other number to load raw MAT: '))

for modulation_number in range(len(modulations)):
    if selection == 1:
        # Filename setup
        pkl_file_name = pathlib.Path(join('../data', str(modulations[modulation_number]) + '_RAW.pickle'))
        #pkl_file_name = 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\data\\' + \
        #                modulations[modulation_number] + \
        #                '_RAW.pickle'

        # Load the pickle file
        with open(pkl_file_name, 'rb') as handle:
            data = pickle.load(handle)
        print(pkl_file_name + ' file loaded...')

        # Quick code to separate SNR
        start = 0
        end = 4096
        signal = np.empty([number_of_snr, number_of_frames, frame_size, 2])
        for snr in range(number_of_snr):
            signal[snr, 0:4096, :, :] = data[0][start:end, :, :]
            start += 4096
            end += 4096
        print('Signal split in different SNR...')

        # Parse signal
        parsed_signal = np.zeros((number_of_snr, number_of_frames, frame_size), dtype=np.complex)
        for snr in range(number_of_snr):
            for frames in range(number_of_frames):
                for samples in range(frame_size):
                    parsed_signal[snr, frames, samples] = (complex(signal[snr, frames, samples, 0],
                                                                   signal[snr, frames, samples, 1]))
        print('Signal parsed...')

        # Calculate features
        features = np.zeros((number_of_snr, number_of_frames, number_of_features))
        for snr in range(number_of_snr):
            for frames in range(number_of_frames):
                print('Calculating for SNR = {0}'.format(snr))
                features[snr, frames, :] = ft.calculate_features(parsed_signal[snr, frames, :])
        print('Features calculated...')

        # Save the samples ina pickle file
        with open(pathlib.Path(join('../data', str(modulations[modulation_number]) + '_features_from_PKL.pickle'), 'wb')) as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved...')
        print('Finished.')
    else:
        # Filename setup
        mat_file_name = pathlib.Path(join('../data', str(modulations[modulation_number]) + '_RAW.mat'))
        #mat_file_name = 'C:\\Users\\ronny\\PycharmProjects\\amcpy\\data\\' + \
        #                modulations[modulation_number] + \
        #                '_RAW.mat'

        # Dictionary to access variable inside MAT file
        info = {'BPSK': 'pks2_signal',
                'QPSK': 'pks4_signal',
                '16QAM': 'qam16_signal'}

        # Load MAT file
        data_mat = scipy.io.loadmat(mat_file_name)
        print(mat_file_name + ' file loaded and already parsed...')
        parsed_signal = data_mat[info[modulations[modulation_number]]]

        # Calculate features
        features = np.zeros((number_of_snr, number_of_frames, number_of_features))
        for snr in range(number_of_snr):
            for frames in range(number_of_frames):
                features[snr, frames, :] = ft.calculate_features(parsed_signal[snr, frames, :])
        print('Features calculated...')

        # Save the samples ina pickle file
        with open(pathlib.Path(join('../data', str(modulations[modulation_number]) + '_features_from_MAT.pickle'), 'wb')) as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved...')
        print('Finished.')
