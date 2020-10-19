import json
import os
import pathlib
import struct
import time
from os.path import join

import numpy as np
import scipy.io
import serial

import features
import functions


def receive_data(port: serial.Serial()) -> (float, int):
    # print('Receiving data...')
    data = []
    start = 0
    while True:
        a = port.read(size=4)
        if a == b'\xCA\xCA\xCA\xCA':
            # print('Head')
            start = 1
        elif a == b'\xF0\xF0\xF0\xF0':
            # print('Tail')
            break
        elif start == 1:
            data.append(a)

    # print('Data length: {}'.format(len(data)))
    if len(data) == 44:
        num_array = np.zeros((22, ), dtype=np.float32)
        counter_array = np.zeros((22, ), dtype=np.int32)
    else:
        num_array = np.zeros((len(data) - 1,), dtype=np.float32)
        counter_array = np.zeros((1,), dtype=np.int32)
    if len(data) == 2:  # Prediction
        print('Prediction')
        new_data = b''.join(data)
        num_array = struct.unpack('<f', new_data[0:4])
        counter_array = struct.unpack('<i', new_data[4:8])
    elif len(data) == 44:  # Features and counters
        print('Features')
        new_data = b''.join(data)
        for x in range(0, 22):
            aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
            num_array[x] = aux[0]
        for x in range(22, 44):
            aux = struct.unpack('<i', new_data[x * 4:x * 4 + 4])
            counter_array[x - 22] = aux[0]
    elif len(data) == frameSize + 1:  # Data
        print('Inst values')
        new_data = b''.join(data)
        for x in range(0, frameSize):
            aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
            num_array[x] = aux[0]
        counter_array = struct.unpack('<i', new_data[frameSize * 4:frameSize * 4 + 4])
    elif len(data) > frameSize + 1:  # Echo
        print('Echo')
        new_data = b''.join(data)
        for x in range(0, frameSize * 8, 8):
            r = struct.unpack('<f', new_data[x:x + 4])
            i = struct.unpack('<f', new_data[x + 4:x + 8])
            real[x // 8] = r[0]
            imag[x // 8] = i[0]
        num_array = None
        counter_array = None
    return num_array, counter_array


isTest = 0

with open("./info.json") as handle:
    info_json = json.load(handle)

modulations = info_json['modulations']['names']
frameSize = info_json['frameSize']

# Filename setup
mat_file_name = pathlib.Path(join(os.getcwd(), 'mat-data', 'testData.mat'))

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
parsed_BPSK_signal = data_mat[info[modulations[0]]]

err_abs_vector = []
err_phase_vector = []
err_unwrapped_phase_vector = []
err_freq_vector = []
err_cn_abs_vector = []
err_features = []
gathered_data = []

if isTest == 0:
    # Setup UART COM on Windows
    ser = serial.Serial(port='COM3', baudrate=115200, parity='N', bytesize=8, stopbits=1, timeout=1)
    real = np.zeros((2048,), dtype=np.float32)
    imag = np.zeros((2048,), dtype=np.float32)
    snr = 0
    for frame in range(0, 500):
        i_values = functions.InstValues(parsed_BPSK_signal[snr, frame, 0:frameSize])
        m_values = functions.MomentValues(parsed_BPSK_signal[snr, frame, 0:frameSize])
        ft = np.float32(features.calculate_features(parsed_BPSK_signal[snr, frame, 0:frameSize]))
        # Write to UART
        print('Transmitting...')
        for point in range(0, 2048):
            binary = struct.pack('<f', np.real(parsed_BPSK_signal[snr, frame, point]))
            ser.write(binary)
            binary = struct.pack('<f', np.imag(parsed_BPSK_signal[snr, frame, point]))
            ser.write(binary)

        received_list = []
        for results in range(0, 8):
            received_list.append(receive_data(ser))

        err = False
        for n in range(0, 2048):
            if abs(real[n] - np.real(parsed_BPSK_signal[snr, frame, n])) > 0:
                err = True
                print('Error at real sample number {}'.format(n))
            if abs(imag[n] - np.imag(parsed_BPSK_signal[snr, frame, n])) > 0:
                err = True
                print('Error at real sample number {}'.format(n))

        if not err:
            print('Echo ok')

        err_abs_vector.append(i_values.inst_abs.T - received_list[1][0])
        print('Instantaneous absolute max error: {:.3}'.format(np.max(np.abs(err_abs_vector))))
        print('Calculation time in clock cycles: {}'.format(received_list[1][1][0]))
        print('Calculation time in ms: {:.3}'.format(received_list[1][1][0] / 200000))
        # plt.plot(err_abs_vector)
        # plt.show()

        err_phase_vector.append(i_values.inst_phase.T - received_list[2][0])
        print('Instantaneous phase max error: {:.3}'.format(np.max(np.abs(err_phase_vector))))
        print('Calculation time in clock cycles: {}'.format(received_list[2][1][0]))
        print('Calculation time in ms: {:.3}'.format(received_list[2][1][0] / 200000))
        # plt.plot(err_phase_vector)
        # plt.show()

        err_unwrapped_phase_vector.append(i_values.inst_unwrapped_phase.T - received_list[3][0])
        print('Instantaneous unwrapped phase max error: {:.3}'.format(np.max(np.abs(err_unwrapped_phase_vector))))
        print('Calculation time in clock cycles: {}'.format(received_list[3][1][0]))
        print('Calculation time in ms: {:.3}'.format(received_list[3][1][0] / 200000))
        # plt.plot(err_unwrapped_phase_vector)
        # plt.show()

        err_freq_vector.append(i_values.inst_freq[0:frameSize - 1].T - received_list[4][0][0:frameSize - 1])
        print('Instantaneous frequency max error: {:.3}'.format(np.max(np.abs(err_freq_vector))))
        print('Calculation time in clock cycles: {}'.format(received_list[4][1][0]))
        print('Calculation time in ms: {:.3}'.format(received_list[4][1][0] / 200000))
        # plt.plot(err_freq_vector)
        # plt.show()

        err_cn_abs_vector.append(i_values.inst_cna.T - received_list[5][0])
        print('Instantaneous CN amplitude max error: {:.3}'.format(np.max(np.abs(err_cn_abs_vector))))
        print('Calculation time in clock cycles: {}'.format(received_list[5][1][0]))
        print('Calculation time in ms: {:.3}'.format(received_list[5][1][0] / 200000))
        # plt.plot(err_cn_abs_vector)
        # plt.show()

        err_features.append(ft - received_list[6][0])
        print('Error list: {}'.format(err_features))
        print('Timings list: {}'.format(received_list[6][1]))
        print('Timings list (ms): {}'.format(received_list[6][1]/200000))

        print('Wait...')
        gathered_data.append(received_list)
        time.sleep(3)
