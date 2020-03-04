import os
import pathlib
import pickle
import threading
import time
from multiprocessing import Process
from os.path import join
from queue import Queue

import numpy as np
import scipy.io

import features as ft

# Global variables config
num_horses = 12
frame_size = 1024
number_of_frames = 1000
number_of_snr = 21
number_of_features = 22
modulations = ['BPSK', 'QPSK', 'PSK8', 'QAM16']


def modulation_process(modulation, selection):
    print('Starting new process...')
    features = np.zeros((number_of_snr, number_of_frames, number_of_features))

    def go_horse():
        snr_array = np.linspace(-20, 30, 26)
        while True:
            item = q.get()
            if item is None:
                break
            features[item[1], item[2], :] = ft.calculate_features(item[0])
            if item[2] == number_of_frames - 1:
                print('Task done for SNR = {0} - Modulation = {1} - Process ID = {2}'.format(snr_array[item[1]],
                                                                                             modulation,
                                                                                             os.getpid()))
            q.task_done()

    if selection == 1:
        # Filename setup
        pkl_file_name = pathlib.Path(join(os.getcwd(), 'data', modulation + '_RAW.pickle'))

        # Load the pickle file
        with open(pkl_file_name, 'rb') as handle:
            data = pickle.load(handle)
        print(str(pkl_file_name) + ' file loaded...')

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

        # Threads setup
        q = Queue()
        threads = []
        for i in range(num_horses):
            horses = threading.Thread(target=go_horse)
            horses.start()
            threads.append(horses)
        print('Threads started...')

        # Calculate features
        for snr in range(number_of_snr):
            for frames in range(number_of_frames):
                q.put([parsed_signal[snr, frames, :], snr, frames])
        q.join()
        print('Features calculated...')

        # Stop workers
        for i in range(num_horses):
            q.put(None)
        for horses in threads:
            horses.join()
        print('Horses stopped...')

        # Save the samples ina pickle file
        with open(pathlib.Path(join(os.getcwd(), 'data', modulation + '_features_from_MAT.pickle')), 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved...')
    elif selection == 2:
        # Filename setup
        mat_file_name = pathlib.Path(join(os.getcwd(), 'data', modulation + '_RAW.mat'))

        # Dictionary to access variable inside MAT file
        info = {'BPSK': 'pks2_signal',
                'QPSK': 'pks4_signal',
                'PSK8': 'pks8_signal',
                'QAM16': 'qam16_signal'}

        # Load MAT file
        data_mat = scipy.io.loadmat(mat_file_name)
        print(str(mat_file_name) + ' file loaded and already parsed...')
        parsed_signal = data_mat[info[modulation]]

        # Threads setup
        q = Queue()
        threads = []
        for i in range(num_horses):
            horses = threading.Thread(target=go_horse)
            horses.start()
            threads.append(horses)
        print('Threads started...')

        # Calculate features
        for snr in range(number_of_snr):
            for frames in range(number_of_frames):
                q.put([parsed_signal[snr, frames, :], snr, frames])
        q.join()
        print('Features calculated...')

        # Stop workers
        for i in range(num_horses):
            q.put(None)
        for horses in threads:
            horses.join()
        print('Horses stopped...')

        # Save the samples in a pickle file
        with open(pathlib.Path(join(os.getcwd(), 'data', modulation + '_features_from_MAT.pickle')), 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved...')
    else:
        # Filename setup
        gr_file_name = pathlib.Path(join(os.getcwd(), 'gr-data', "pickle", modulation + '.pickle'))

        # Load the pickle file
        with open(gr_file_name, 'rb') as handle:
            dataRaw = pickle.load(handle)
        print(str(gr_file_name) + ' file loaded...')

        print("Spliting data from GR...")

        data = np.zeros((len(dataRaw), number_of_frames, frame_size), dtype=np.complex64)        
        for snr in range(len(dataRaw)):
            data[snr][:] = np.split(dataRaw[snr][:number_of_frames*frame_size], number_of_frames)

        print("{} data splitted into {} frames containing {} symbols.".format(modulation, number_of_frames, frame_size))
        
        for snr in range(len(data)):
            for frame in range(len(data[snr])):
                features[snr][frame][:] = ft.calculate_features(data[snr][frame][:])

        with open(pathlib.Path(join(os.getcwd(), "gr-data", "pickle", str(modulation) + "_features.pickle")), 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Process time in seconds: {0}'.format(time.process_time()))
    print('Done.')

if __name__ == '__main__':
    slaves = []
    for mod in modulations:
        new_slave = Process(target=modulation_process, args=(mod, 3))
        slaves.append(new_slave)

    for i in range(len(modulations)):
        slaves[i].start()

    for i in range(len(modulations)):
        slaves[i].join()

    print('Lord is happy now, job is done!')
