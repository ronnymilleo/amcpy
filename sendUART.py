import json
import os
import pathlib
import struct
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import serial

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

    for results in range(0, 5):
        # Create string to receive echo data
        string = ""

        print('Receiving timings...')
        # Wait for data to be returned
        while True:
            char = ser.read().decode('utf-8', 'ignore')
            string = string + char
            if char == '&':  # End of transmission char (by Ronny)
                break
        print(string)

        print('Receiving data...')
        data = []
        start = 0
        while True:
            a = ser.read(size=4)
            if a == b'\xCA\xCA\xCA\xCA':
                print('Head')
                start = 1
            elif a == b'\xF0\xF0\xF0\xF0':
                print('Tail')
                break
            elif start == 1:
                data.append(a)

        received = []
        if len(data) == 2048:  # Echo
            new_data = b''.join(data)
            real = []
            imag = []

            for x in range(8, 8200, 8):
                real.append(struct.unpack('<f', new_data[x - 8:x - 4]))
                imag.append(struct.unpack('<f', new_data[x - 4:x]))

            real_n = np.array(real, dtype=np.float32)
            imag_n = np.array(imag, dtype=np.float32)
            err_real = np.ones(1024)
            err_imag = np.ones(1024)

            # Errors
            for i in range(0, 1024):
                err_real[i] = np.real(parsed_BPSK_signal[0, 0, i]) - real_n[i]
                err_imag[i] = np.imag(parsed_BPSK_signal[0, 0, i]) - imag_n[i]
        else:  # Data
            new_data = b''.join(data)
            for x in range(0, 4096, 4):
                received.append(struct.unpack('<f', new_data[x:x + 4]))

            if results == 0:
                rx_inst_abs = np.array(received, dtype=np.float32)
                plt.figure(num=0, figsize=(6.4, 3.6), dpi=300)
                plt.plot(inst_abs)
                plt.plot(rx_inst_abs, '--')
                plt.show()
            elif results == 1:
                rx_inst_phase = np.array(received, dtype=np.float32)
                plt.figure(num=1, figsize=(6.4, 3.6), dpi=300)
                plt.plot(inst_phase)
                plt.plot(rx_inst_phase, '--')
                plt.show()
            elif results == 2:
                rx_inst_unwrapped_phase = np.array(received, dtype=np.float32)
                plt.figure(num=2, figsize=(6.4, 3.6), dpi=300)
                plt.plot(inst_unwrapped_phase)
                plt.plot(rx_inst_unwrapped_phase, '--')
                plt.show()
            elif results == 3:
                rx_inst_freq = np.array(received, dtype=np.float32)
                plt.figure(num=3, figsize=(6.4, 3.6), dpi=300)
                plt.plot(inst_freq)
                plt.plot(rx_inst_freq, '--')
                plt.show()
            elif results == 4:
                rx_inst_cn_abs = np.array(received, dtype=np.float32)
                plt.figure(num=4, figsize=(6.4, 3.6), dpi=300)
                plt.plot(inst_cn_abs)
                plt.plot(rx_inst_cn_abs, '--')
                plt.show()

    err_abs_vector = inst_abs - rx_inst_abs
    print(np.max(err_abs_vector))
    plt.plot(err_abs_vector)
    plt.show()

    err_phase_vector = inst_phase - rx_inst_phase
    print(np.max(err_phase_vector))
    plt.plot(err_phase_vector)
    plt.show()

    err_unwrapped_phase_vector = inst_unwrapped_phase - rx_inst_unwrapped_phase
    print(np.max(err_unwrapped_phase_vector))
    plt.plot(err_unwrapped_phase_vector)
    plt.show()

    err_freq_vector = inst_freq[0:1023] - rx_inst_freq[0:1023]
    print(np.max(err_freq_vector))
    plt.plot(err_freq_vector)
    plt.show()

    err_cn_abs_vector = inst_cn_abs - rx_inst_cn_abs
    print(np.max(err_cn_abs_vector))
    plt.plot(err_cn_abs_vector)
    plt.show()
