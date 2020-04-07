import pathlib
from os.path import join
import json

import gif
import matplotlib.pyplot as plt
import numpy as np


@gif.frame
def plot_time_gif(x, y, start, end, step):
    plt.figure(figsize=(14, 6), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(x + step, np.real(y[start + step:end + step]))
    plt.plot(x + step, np.imag(y[start + step:end + step]))
    plt.axis([x[0] + step, x[-1] + step, -4, 4])
    plt.title('Frame {}'.format(step // frame_size))
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(y[start + step:end + step]), np.imag(y[start + step:end + step]))
    plt.title('Frame {}'.format(step // frame_size))


# CONFIG
with open("./info.json") as handle:
    info_json = json.load(handle)

snr = 20
gif_frame_duration = int(1000 / 25)  # milliseconds
frame_size = 1024
number_of_frames = 800
modulations = info_json['modulations']['names']
data_set = info_json['dataSetForTraining']

for modulation in modulations:
    if data_set == "rayleigh":
        file_name = pathlib.Path(
            join('gr-data', 'binary', 'binary_' + modulation + "(" + "{}".format(snr) + ")"))
    if data_set == "awgn":
        file_name = pathlib.Path(
            join('gr-data', 'binary', 'binary_awgn_' + modulation + "(" + "{}".format(snr) + ")"))
    
    data = np.fromfile(file_name, dtype=np.complex64)

    frames = []
    init_start = 300 * 8  # initial delay
    init_end = init_start + frame_size
    init_x = np.linspace(init_start, init_end, frame_size)
    for new_step in range(0, frame_size * number_of_frames, frame_size):
        frame = plot_time_gif(init_x, data, init_start, init_end, new_step)
        frames.append(frame)
        print(modulation + ' {} frames appended...'.format(new_step // frame_size))

    if data_set == "rayleigh":
        gif.save(frames, modulation + '.gif', duration=gif_frame_duration)
    if data_set == "awgn":
        gif.save(frames, modulation + '_awgn.gif', duration=gif_frame_duration)