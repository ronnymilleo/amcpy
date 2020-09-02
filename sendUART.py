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
inst_freq = np.transpose(
    np.array(1 / (2 * np.pi) * np.diff(np.unwrap(np.angle(parsed_BPSK_signal[0, 0, 0:1024]))), dtype=np.float32,
             ndmin=2))

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

    # Create string to receive echo data
    string = ""

    print('Receiving timings...')
    # Wait for data to be returned
    while True:
        char = ser.read().decode(encoding="utf-8")
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

        received = np.array(received, dtype=np.float32)
        plt.plot(received[0:128])
        plt.show()

    err_abs_vector = inst_abs - received
    plt.plot(err_abs_vector[0:128])
    plt.show()
    err_phase_vector = inst_phase - received
    #  err_freq_vector = inst_freq - received
