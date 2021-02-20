import time
from multiprocessing import Process
from queue import Queue
from threading import Thread

import scipy.io

import features as ft
from globals import *

number_of_frames = number_of_testing_frames
number_of_SNR = len(testing_SNR)
features = np.zeros((number_of_SNR, number_of_frames, len(used_features)), dtype=np.float32)


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
    for SNR in range(number_of_SNR):  # Every SNR
        for frame in range(number_of_frames):  # of every frame will be at the Queue waiting to be calculated
            queue.put([parsed_signal[SNR, frame, 0:frame_size], SNR, frame])  # Run!
    queue.join()  # This is the line that synchronizes everything
    print('Features calculated...')

    # Save the samples in a mat file
    save_dict = {'Modulation': modulation, mat_info[modulation]: features}
    scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'mat-data', modulation + '_best_features.mat')), save_dict)

    print('File saved...')
    print('Process time in seconds: {0}'.format(time.process_time()))  # Horses benchmark!
    print('Done.')


if __name__ == '__main__':
    processes = []
    for mod in signals:  # Create a process for each modulation (6 processes)
        new_process = Process(target=modulation_process, args=(mod,))
        processes.append(new_process)

    for i in range(len(signals)):
        processes[i].start()
        processes[i].join()

    print('Finish')
