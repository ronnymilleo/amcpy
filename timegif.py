import json
import pathlib
from os.path import join

import gif
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


@gif.frame
def plot_time_gif(x, y, start, end, step):
    plt.figure(figsize=(14, 6), dpi=150)
    plt.subplot(1, 2, 1)
    if data_set == "rayleigh":
        plt.plot(x, np.real(y[start + step:end + step]))
        plt.plot(x, np.imag(y[start + step:end + step]))
    else:
        plt.plot(x, np.real(y[start:end]))
        plt.plot(x, np.imag(y[start:end]))
    plt.axis([x[0], x[-1], -4, 4])
    plt.title('Frame {}'.format(step // frame_size))
    plt.subplot(1, 2, 2)
    if data_set == "rayleigh":
        plt.scatter(np.real(y[start + step:end + step]), np.imag(y[start + step:end + step]))
    else:
        plt.scatter(np.real(y[start:end]), np.imag(y[start:end]))
    plt.title('Frame {}'.format(step // frame_size))


# CONFIG
with open("./info.json") as handle:
    info_json = json.load(handle)

snr = 20
gif_frame_duration = int(1000 / 2)  # milliseconds
frame_size = 1024
number_of_frames = 100
modulations = info_json['modulations']['names']
data_set = info_json['dataSetForTraining']

for modulation in modulations:
    data = []
    if data_set == "rayleigh":
        file_name = pathlib.Path(
            join('gr-data', 'binary', 'binary_' + modulation + "(" + "{}".format(snr) + ")"))
        data = np.fromfile(file_name, dtype=np.complex64)

    elif data_set == "matlab":
        file_name = pathlib.Path(
            join('mat-data', modulation + '.mat'))
        info = {'BPSK': 'signal_bpsk',
                'QPSK': 'signal_qpsk',
                'PSK8': 'signal_8psk',
                'QAM16': 'signal_qam16',
                'QAM64': 'signal_qam64',
                'noise': 'signal_noise'}
        data_mat = scipy.io.loadmat(file_name)
        print(str(file_name) + ' file loaded...')
        data = data_mat[info[modulation]]

    frames = []
    snr_id = 15  # 20 dB
    init_start = 0  # initial delay
    init_end = init_start + frame_size
    init_x = np.linspace(init_start, init_end, frame_size)

    if data_set == "matlab":
        for new_frame in range(0, number_of_frames):
            frame = plot_time_gif(init_x, data[snr_id][new_frame][0:frame_size], 0, frame_size, new_frame * 1024)
            frames.append(frame)
            print(modulation + ' {} frames appended...'.format(new_frame + 1))
    else:
        for new_step in range(0, frame_size * number_of_frames, frame_size):
            frame = plot_time_gif(init_x, data, init_start, init_end, new_step)
            frames.append(frame)
            print(modulation + ' {} frames appended...'.format(new_step // frame_size))

    if data_set == "rayleigh":
        gif.save(frames, modulation + '.gif', duration=gif_frame_duration)
    if data_set == "matlab":
        gif.save(frames, modulation + '_matlab.gif', duration=gif_frame_duration)
