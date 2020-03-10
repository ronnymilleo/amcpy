import pathlib
from os.path import join

import gif
import matplotlib.pyplot as plt
import numpy as np


@gif.frame
def plot_time_gif(x, y, start, end, step):
    plt.figure(figsize=(6.4, 3.6), dpi=100)
    plt.plot(x+step, np.real(y[start+step:end+step]))
    plt.plot(x+step, np.imag(y[start+step:end+step]))
    plt.axis([x[0]+step, x[-1]+step, -4, 4])


snr = 20
modulations = ['BPSK', 'QPSK', 'PSK8', 'QAM16']
for modulation in modulations:
    file_name = pathlib.Path(
        join('local-test-data', 'binary_' + modulation + "(" + "{}".format(snr) + ")"))

    data = np.fromfile(file_name, dtype=np.complex64)

    frames = []
    init_start = 256
    init_end = init_start + 1024
    init_x = np.linspace(init_start, init_end, 1024)
    for new_step in range(0, 10000, 10):
        frame = plot_time_gif(init_x, data, init_start, init_end, new_step)
        frames.append(frame)
        print(modulation + ' {} frame appended...'.format(new_step))

    gif.save(frames, modulation + '.gif', duration=1/24)
