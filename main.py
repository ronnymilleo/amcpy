import json
import os
import pathlib
import time
from multiprocessing import Process
from os.path import join
from queue import Queue
from threading import Thread

import numpy as np
import scipy.io

import features as ft

with open("./info.json") as handle:
    info_json = json.load(handle)

num_threads = 2
modulations = info_json['modulations']['names']
data_set = info_json['dataSetForTraining']
frame_size = info_json['frameSize']
snr_index = np.array(info_json['snr']['using'])
snr_dB_array = (snr_index - 10) * 2
nb_of_snr = len(info_json['snr']['using'])
nb_of_frames = info_json['numberOfFrames']
nb_of_features = len(info_json['features']['using'])
features = np.zeros((nb_of_snr, nb_of_frames, nb_of_features))


class Worker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            item = self.queue.get()
            try:
                features[item[1], item[2], :] = ft.calculate_features(item[0])
            finally:
                self.queue.task_done()


def modulation_process(modulation):
    print('Starting new process...')

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

    # Threads setup
    queue = Queue()

    for x in range(num_threads):
        worker = Worker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
        print('Starting new thread...')

    # Calculate features
    for snr in range(nb_of_snr):  # Every SNR
        for frames in range(nb_of_frames):  # of every frame will be at the Queue waiting to be calculated
            queue.put([parsed_signal[snr, frames, 0:frame_size], snr, frames])  # Run!
    queue.join()  # This is the line that synchronizes everything
    print('Features calculated...')

    # Save the samples in a mat file
    save_dict = {'Modulation': modulation, info[modulation]: features}
    scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'mat-data', modulation + '_features.mat')), save_dict)

    print('File saved...')
    print('Process time in seconds: {0}'.format(time.process_time()))  # Horses benchmark!
    print('Done.')


if __name__ == '__main__':
    processes = []
    for mod in modulations:  # Create a process for each modulation (6 processes)
        new_process = Process(target=modulation_process, args=(mod,))
        processes.append(new_process)

    for i in range(len(modulations)):
        processes[i].start()
        processes[i].join()

    print('Finish')
