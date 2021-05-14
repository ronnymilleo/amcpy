import os
import pathlib
import time
from multiprocessing import Process
from os.path import join
from queue import Queue
from threading import Thread

import scipy.io

import functions
from globals import mat_info, num_threads, used_features, features_matrix, frame_size, signals


class Worker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            item = self.queue.get()
            try:
                features_matrix[item[1], item[2], :] = functions.calculate_features(used_features, item[0])
            finally:
                self.queue.task_done()


def modulation_process(modulation: str):
    print('Starting new process...')

    # Filename setup
    mat_file_name = pathlib.Path(join(os.getcwd(), 'mat-data', 'all_modulations.mat'))

    # Load MAT file
    data_mat = scipy.io.loadmat(mat_file_name)
    print(str(mat_file_name) + ' file loaded...')
    parsed_signal = data_mat[mat_info[modulation]]

    # Threads setup
    queue = Queue()

    for x in range(num_threads):
        worker = Worker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
        print('Starting new thread...')

    # Calculate features
    for snr in range(len(features_matrix)):  # Every snr
        for frame in range(len(features_matrix[0])):  # of every frame will be at the Queue waiting to be calculated
            queue.put([parsed_signal[snr, frame, 0:frame_size], snr, frame])  # Run!
    queue.join()  # This is the line that synchronizes everything
    print('Features calculated...')

    # Save the samples in a mat file
    save_dict = {'Modulation': modulation, mat_info[modulation]: features_matrix}
    scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'calculated-features', modulation + '_features.mat')), save_dict)
    print('Process time in seconds: {0}'.format(time.process_time()))  # Horses benchmark!
    print('Done.')


def run():
    processes = []
    for mod in signals.values():  # Create a process for each modulation (6 processes)
        new_process = Process(target=modulation_process, args=(mod,))
        processes.append(new_process)

    for i in range(len(signals)):
        processes[i].start()
        processes[i].join()
