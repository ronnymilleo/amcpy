import time
from multiprocessing import Process
from queue import Queue
from threading import Thread

import scipy.io

import functions
from globals import *


class Worker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            item = self.queue.get()
            try:
                features_matrix[item[1], item[2], :] = functions.calculate_features(all_features, item[0])
            finally:
                self.queue.task_done()


def modulation_process(modulation: str):
    print('Starting new process...')

    # Load MAT file
    mat_file_name = matlab_data_folder.joinpath(matlab_data_filename).__str__()
    data_mat = scipy.io.loadmat(mat_file_name)
    print(str(mat_file_name) + ' file loaded...')
    parsed_signal = data_mat[mat_info[modulation]]

    # Threads setup
    queue = Queue()

    for _ in range(num_threads):
        worker = Worker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
        print('Starting new thread...')

    # Calculate features
    for snr in range(0, number_of_snr):  # Do it for every SNR
        for frame in range(0, number_of_frames):  # of every frame
            queue.put([parsed_signal[snr, frame, 0:frame_size], snr, frame])  # Run!
    queue.join()  # This is the line that synchronizes everything
    print('Features calculated...')

    # Save the samples in a mat file
    save_dict = {'Modulation': modulation, mat_info[modulation]: features_matrix}
    scipy.io.savemat(calculated_features_folder.joinpath(modulation + '_features.mat'), save_dict)
    print('Process time in seconds: {0}'.format(time.process_time()))  # Workers benchmark!
    print('Done.')


def run():
    processes = []
    for mod in modulation_signals_with_noise:  # Create a process for each modulation + noise (6 processes)
        new_process = Process(target=modulation_process, args=(mod,))
        processes.append(new_process)

    for i in range(0, len(modulation_signals_with_noise)):
        processes[i].start()
        processes[i].join()

    print('Features calculations done!')
