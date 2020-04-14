import json
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

with open("./info.json") as handle:
    info_json = json.load(handle)

num_horses = 2  # Well, that's should be your horsepower... be careful
frame_size = info_json['frameSize']
nb_of_frames = info_json['numberOfFrames']
nb_of_snr = len(info_json['snr']['using'])
nb_of_features = len(info_json['features']['using'])
modulations = info_json['modulations']['names']
data_set = info_json['dataSetForTraining']


def modulation_process(modulation, selection):
    print('Starting new process...')
    features = np.zeros((nb_of_snr, nb_of_frames, nb_of_features))

    # Function to be threaded and processed in parallel
    def go_horse():
        # snr_array = np.linspace(-20, 20, 21)  # Let's make sure we're getting only the necessary SNR
        snr_array = list(map(int, [info_json['snr']['values'][i] for i in info_json['snr']['using']]))
        while True:
            item = q.get()  # This line gets values from queue to evaluate
            if item is None:  # This line will run only when queue is empty (job done)
                break
            features[item[1], item[2], :] = ft.calculate_features(item[0])
            if item[2] == nb_of_frames - 1:
                print('Task done for SNR = {0} - Modulation = {1} - Process ID = {2}'.format(snr_array[item[1]],
                                                                                             modulation,
                                                                                             os.getpid()))
            q.task_done()  # This line says "hey, I'm done, give-me more!"

    # Selection will define where the data is coming from and how it will be saved after
    if selection == 1:  # Pickle broken dataset from "Over-the-air bla bla bla" paper
        # Filename setup
        pkl_file_name = pathlib.Path(join(os.getcwd(), 'data', modulation + '_RAW.pickle'))

        # Load the pickle file
        with open(pkl_file_name, 'rb') as pkl_handle:  # Pickle handle -> renamed to not shadow handle from outer scope
            data_pkl = pickle.load(pkl_handle)
        print(str(pkl_file_name) + ' file loaded...')

        # Quick code to separate SNR
        start = 0
        end = 4096
        signal = np.empty([nb_of_snr, nb_of_frames, frame_size, 2])
        for snr in range(nb_of_snr):
            signal[snr, 0:4096, :, :] = data_pkl[0][start:end, :, :]
            start += 4096
            end += 4096
        print('Signal split in different SNR...')

        # Parse signal
        parsed_signal = np.zeros((nb_of_snr, nb_of_frames, frame_size), dtype=np.complex64)
        for snr in range(nb_of_snr):
            for frames in range(nb_of_frames):
                for samples in range(frame_size):
                    parsed_signal[snr, frames, samples] = (complex(signal[snr, frames, samples, 0],
                                                                   signal[snr, frames, samples, 1]))
        print('Signal parsed...')
    elif selection == 2:  # MATLAB generated dataset with no Rayleigh ='( (but it's de most reliable: updated 10/04)
        # Filename setup
        mat_file_name = pathlib.Path(join(os.getcwd(), 'mat-data', modulation + '.mat'))

        # Dictionary to access variable inside MAT file
        info = {'BPSK': 'signal_bpsk',
                'QPSK': 'signal_qpsk',
                'PSK8': 'signal_8psk',
                'QAM16': 'signal_qam16',
                'QAM64': 'signal_qam64',
                'noise': 'signal_noise'}

        # Load MAT file
        data_mat = scipy.io.loadmat(mat_file_name)
        print(str(mat_file_name) + ' file loaded...')
        parsed_signal = data_mat[info[modulation]]
        print('Signal parsed...')
    else:  # GNURadio generated dataset -- the best ever (not actually: updated 10/04)
        # Filename setup
        try:
            if data_set == "rayleigh":
                gr_file_name = pathlib.Path(join(os.getcwd(), 'gr-data', "pickle", modulation + '.pickle'))
                # Load the pickle file
                with open(gr_file_name, 'rb') as gr_handle:  # Same as said at selection 1 -- duh
                    data_gr = pickle.load(gr_handle)
                print(str(gr_file_name) + ' file loaded...')
            if data_set == "awgn":
                gr_file_name = pathlib.Path(join(os.getcwd(), 'gr-data', "pickle", modulation + '_awgn.pickle'))
                # Load the pickle file
                with open(gr_file_name, 'rb') as gr_handle:  # Same as said at selection 1 -- duh
                    data_gr = pickle.load(gr_handle)
                print(str(gr_file_name) + ' file loaded...')
        except FileNotFoundError:
            if data_set == "rayleigh":
                gr_file_name = pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                                 'gr-data',
                                                 "pickle",
                                                 modulation + '.pickle'))
                with open(gr_file_name, 'rb') as gr_handle:  # Same as said at selection 1 -- duh
                    data_gr = pickle.load(gr_handle)
                print(str(gr_file_name) + ' file loaded...')
            if data_set == "awgn":
                gr_file_name = pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                                 'gr-data',
                                                 "pickle",
                                                 modulation + '_awgn.pickle'))
                with open(gr_file_name, 'rb') as gr_handle:  # Same as said at selection 1 -- duh
                    data_gr = pickle.load(gr_handle)
                print(str(gr_file_name) + ' file loaded...')
        # Parsing signal
        print("Splitting data from GR...")
        parsed_signal = np.zeros((len(data_gr), nb_of_frames, frame_size), dtype=np.complex64)
        for i, snr in enumerate(info_json['snr']['using']):
            parsed_signal[i][:] = np.split(data_gr[snr][1024:(nb_of_frames * frame_size) + 1024], nb_of_frames)
        print("{} data split into {} frames containing {} symbols.".format(modulation, nb_of_frames, frame_size))

    # Calculating features using threads...
    # Threads setup
    q = Queue()
    threads = []
    for _ in range(num_horses):
        horses = threading.Thread(target=go_horse)
        horses.start()  # All threads will be started and will wait for instructions from their master (me)
        threads.append(horses)  # For every time this program run, it will allocate that number of threads
    print('Threads started...')

    # Calculate features
    for snr in range(nb_of_snr):  # Every SNR
        for frames in range(nb_of_frames):  # of every frame wil be at the Queue waiting to be calculated
            q.put([parsed_signal[snr, frames, 0:frame_size], snr, frames])  # Run!
    q.join()  # This is the line that synchronizes everything, so threads that finish first will wait ok?
    print('Features calculated...')

    # Stop workers
    for _ in range(num_horses):
        q.put(None)  # So we put nothing into the Queue to finish everything
    for horses in threads:
        horses.join()  # Let the horses rest now, make sure your computer is cool enough before running again
    print('Horses stopped...')

    if selection == 1:
        # Save the samples in a pickle file
        with open(pathlib.Path(join(os.getcwd(), 'data', modulation + '_features_from_PKL.pickle')), 'wb') as p_handle:
            pickle.dump(features, p_handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved...')
    elif selection == 2:
        # Save the samples in a pickle file
        with open(pathlib.Path(join(os.getcwd(), 'mat-data', modulation + '_features.pickle')), 'wb') as m_handle:
            pickle.dump(features, m_handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved...')
    else:
        try:
            # Save the samples in a pickle file
            if data_set == "rayleigh":
                with open(pathlib.Path(join(os.getcwd(), "gr-data", "pickle", str(modulation) + "_features.pickle")),
                          'wb') as gr_handle:
                    pickle.dump(features, gr_handle, protocol=pickle.HIGHEST_PROTOCOL)
            if data_set == "awgn":
                with open(
                        pathlib.Path(join(os.getcwd(), "gr-data", "pickle", str(modulation) + "_awgn_features.pickle")),
                        'wb') as gr_handle:
                    pickle.dump(features, gr_handle, protocol=pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            # Save the samples in a pickle file
            if data_set == "rayleigh":
                with open(pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                            "gr-data",
                                            "pickle",
                                            str(modulation) + "_features.pickle")),
                          'wb') as gr_handle:
                    pickle.dump(features, gr_handle, protocol=pickle.HIGHEST_PROTOCOL)
            if data_set == "awgn":
                with open(pathlib.Path(join('C:\\Users\\ronny\\Google Drive\\Colab Notebooks',
                                            "gr-data",
                                            "pickle",
                                            str(modulation) + "_awgn_features.pickle")),
                          'wb') as gr_handle:
                    pickle.dump(features, gr_handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved...')

    print('Process time in seconds: {0}'.format(time.process_time()))  # Horses benchmark!
    print('Done.')


if __name__ == '__main__':
    dataset = 2  # Use GNURadio dataset
    brothers = []  # Every brother of mine will have the same horsepower we set before
    for mod in modulations:
        new_slave = Process(target=modulation_process, args=(mod, dataset))
        brothers.append(new_slave)  # Here we have my brothers * horses = total power

    for i in range(len(modulations)):
        brothers[i].start()  # We start together

    for i in range(len(modulations)):
        brothers[i].join()  # We finish together (and wait for the slowest)

    print('Celebrate! We have done the job you asked!')
