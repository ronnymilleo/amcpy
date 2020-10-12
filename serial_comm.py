import json
import os
import pathlib
import struct
from os.path import join

import numpy as np
import scipy.io
import serial


def receive_data(port: serial.Serial()) -> (float, int):
    print('Receiving data...')
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

    print('Data length: {}'.format(len(data)))
    num_array = np.zeros((len(data) - 1,), dtype=np.float32)
    counter = np.zeros((1,), dtype=np.int32)
    if len(data) == 2:  # Feature
        new_data = b''.join(data)
        num_array = struct.unpack('<f', new_data[0:4])
        counter = struct.unpack('<i', new_data[4:8])
    elif len(data) == 1025:  # Data
        new_data = b''.join(data)
        for x in range(0, 1024):
            aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
            num_array[x] = aux[0]
        counter = struct.unpack('<i', new_data[4096:4100])
    elif len(data) == 2048:  # Echo
        new_data = b''.join(data)
        real = np.ndarray((len(data) // 2,), dtype=np.float32)
        imag = np.ndarray((len(data) // 2,), dtype=np.float32)
        for x in range(8, 8200, 8):
            real[x] = struct.unpack('<f', new_data[x - 8:x - 4])
            imag[x] = struct.unpack('<f', new_data[x - 4:x])
    return num_array, counter


isTest = 0

with open("./info.json") as handle:
    info_json = json.load(handle)

modulations = info_json['modulations']['names']

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
parsed_QPSK_signal = data_mat[info[modulations[1]]]
parsed_PSK8_signal = data_mat[info[modulations[2]]]
parsed_QAM16_signal = data_mat[info[modulations[3]]]
parsed_QAM64_signal = data_mat[info[modulations[4]]]
parsed_noise_signal = data_mat[info[modulations[5]]]

inst_abs = np.transpose(np.array(abs(parsed_BPSK_signal[0, 0, 0:1024]), dtype=np.float32, ndmin=2))
inst_phase = np.transpose(np.array(np.angle(parsed_BPSK_signal[0, 0, 0:1024]), dtype=np.float32, ndmin=2))
inst_unwrapped_phase = np.transpose(
    np.array(np.unwrap(np.angle(parsed_BPSK_signal[0, 0, 0:1024])), dtype=np.float32, ndmin=2))
inst_freq = np.transpose(
    np.array(1 / (2 * np.pi) * np.diff(np.unwrap(np.angle(parsed_BPSK_signal[0, 0, 0:1024]))), dtype=np.float32,
             ndmin=2))
inst_cn_abs = (inst_abs / np.mean(inst_abs)) - 1

rx_inst_abs = np.zeros((1024, 1))
rx_inst_freq = np.zeros((1024, 1))
rx_inst_phase = np.zeros((1024, 1))
rx_inst_unwrapped_phase = np.zeros((1024, 1))
rx_inst_cn_abs = np.zeros((1024, 1))

if isTest == 0:
    # Setup UART COM on Windows
    ser = serial.Serial(port='COM3', baudrate=115200, parity='N', bytesize=8, stopbits=1, timeout=1)

    # Write to UART
    print('Transmitting...')
    for point in range(0, 1024):
        binary = struct.pack('<f', np.real(parsed_BPSK_signal[0, 0, point]))
        ser.write(binary)
        binary = struct.pack('<f', np.imag(parsed_BPSK_signal[0, 0, point]))
        ser.write(binary)

    received_list = []
    for results in range(0, 11):
        received_list.append(receive_data(ser))

    err_abs_vector = inst_abs.T - received_list[0][0]
    print('Instantaneous absolute max error: {:.3}'.format(np.max(err_abs_vector)))
    print('Calculation time in clock cycles: {}'.format(received_list[0][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[0][1][0] / 240000))
    # plt.plot(err_abs_vector)
    # plt.show()

    err_phase_vector = inst_phase.T - received_list[1][0]
    print('Instantaneous phase max error: {:.3}'.format(np.max(err_phase_vector)))
    print('Calculation time in clock cycles: {}'.format(received_list[1][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[1][1][0] / 240000))
    # plt.plot(err_phase_vector)
    # plt.show()

    err_unwrapped_phase_vector = inst_unwrapped_phase.T - received_list[2][0]
    print('Instantaneous unwrapped phase max error: {:.3}'.format(np.max(err_unwrapped_phase_vector)))
    print('Calculation time in clock cycles: {}'.format(received_list[2][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[2][1][0] / 240000))
    # plt.plot(err_unwrapped_phase_vector)
    # plt.show()

    err_freq_vector = inst_freq[0:1023].T - received_list[3][0][0:1023]
    print('Instantaneous frequency max error: {:.3}'.format(np.max(err_freq_vector)))
    print('Calculation time in clock cycles: {}'.format(received_list[3][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[3][1][0] / 240000))
    # plt.plot(err_freq_vector)
    # plt.show()

    err_cn_abs_vector = inst_cn_abs.T - received_list[4][0]
    print('Instantaneous CN amplitude max error: {:.3}'.format(np.max(err_cn_abs_vector)))
    print('Calculation time in clock cycles: {}'.format(received_list[4][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[4][1][0] / 240000))
    # plt.plot(err_cn_abs_vector)
    # plt.show()
