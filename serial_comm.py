import json
import os
import pathlib
import struct
from os.path import join

import numpy as np
import scipy.io
import serial

import features
import functions


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
    elif len(data) == frameSize + 1:  # Data
        new_data = b''.join(data)
        for x in range(0, frameSize):
            aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
            num_array[x] = aux[0]
        counter = struct.unpack('<i', new_data[frameSize * 4:frameSize * 4 + 4])
    elif len(data) == frameSize * 2:  # Echo
        new_data = b''.join(data)
        real = np.ndarray((len(data) // 2,), dtype=np.float32)
        imag = np.ndarray((len(data) // 2,), dtype=np.float32)
        for x in range(8, frameSize * 4 + 8, 8):
            real[x] = struct.unpack('<f', new_data[x - 8:x - 4])
            imag[x] = struct.unpack('<f', new_data[x - 4:x])
    return num_array, counter


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
parsed_QPSK_signal = data_mat[info[modulations[1]]]
parsed_PSK8_signal = data_mat[info[modulations[2]]]
parsed_QAM16_signal = data_mat[info[modulations[3]]]
parsed_QAM64_signal = data_mat[info[modulations[4]]]
parsed_noise_signal = data_mat[info[modulations[5]]]

i_values = functions.InstValues(parsed_BPSK_signal[0, 0, 0:frameSize])
m_values = functions.MomentValues(parsed_BPSK_signal[0, 0, 0:frameSize])
ft = np.float32(features.calculate_features(parsed_BPSK_signal[0, 0, 0:frameSize]))

if isTest == 0:
    # Setup UART COM on Windows
    ser = serial.Serial(port='COM3', baudrate=115200, parity='N', bytesize=8, stopbits=1, timeout=1)

    # Write to UART
    print('Transmitting...')
    for point in range(0, 2048):
        binary = struct.pack('<f', np.real(parsed_BPSK_signal[0, 0, point]))
        ser.write(binary)
        binary = struct.pack('<f', np.imag(parsed_BPSK_signal[0, 0, point]))
        ser.write(binary)

    received_list = []
    for results in range(0, 27):
        received_list.append(receive_data(ser))

    err_abs_vector = i_values.inst_abs.T - received_list[0][0]
    print('Instantaneous absolute max error: {:.3}'.format(np.max(np.abs(err_abs_vector))))
    print('Calculation time in clock cycles: {}'.format(received_list[0][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[0][1][0] / 240000))
    # plt.plot(err_abs_vector)
    # plt.show()

    err_phase_vector = i_values.inst_phase.T - received_list[1][0]
    print('Instantaneous phase max error: {:.3}'.format(np.max(np.abs(err_phase_vector))))
    print('Calculation time in clock cycles: {}'.format(received_list[1][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[1][1][0] / 240000))
    # plt.plot(err_phase_vector)
    # plt.show()

    err_unwrapped_phase_vector = i_values.inst_unwrapped_phase.T - received_list[2][0]
    print('Instantaneous unwrapped phase max error: {:.3}'.format(np.max(np.abs(err_unwrapped_phase_vector))))
    print('Calculation time in clock cycles: {}'.format(received_list[2][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[2][1][0] / 240000))
    # plt.plot(err_unwrapped_phase_vector)
    # plt.show()

    err_freq_vector = i_values.inst_freq[0:frameSize - 1].T - received_list[3][0][0:frameSize - 1]
    print('Instantaneous frequency max error: {:.3}'.format(np.max(np.abs(err_freq_vector))))
    print('Calculation time in clock cycles: {}'.format(received_list[3][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[3][1][0] / 240000))
    # plt.plot(err_freq_vector)
    # plt.show()

    err_cn_abs_vector = i_values.inst_cna.T - received_list[4][0]
    print('Instantaneous CN amplitude max error: {:.3}'.format(np.max(np.abs(err_cn_abs_vector))))
    print('Calculation time in clock cycles: {}'.format(received_list[4][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[4][1][0] / 240000))
    # plt.plot(err_cn_abs_vector)
    # plt.show()

    err_ft0 = ft[0] - received_list[5][0]
    print('Feature 0 error: {:.3}'.format(np.max(np.abs(err_ft0))))
    print('Calculation time in clock cycles: {}'.format(received_list[5][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[5][1][0] / 240000))

    err_ft1 = ft[1] - received_list[6][0]
    print('Feature 1 error: {:.3}'.format(np.max(np.abs(err_ft1))))
    print('Calculation time in clock cycles: {}'.format(received_list[6][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[6][1][0] / 240000))

    err_ft2 = ft[2] - received_list[7][0]
    print('Feature 2 error: {:.3}'.format(np.max(np.abs(err_ft2))))
    print('Calculation time in clock cycles: {}'.format(received_list[7][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[7][1][0] / 240000))

    err_ft3 = ft[3] - received_list[8][0]
    print('Feature 3 error: {:.3}'.format(np.max(np.abs(err_ft3))))
    print('Calculation time in clock cycles: {}'.format(received_list[8][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[8][1][0] / 240000))

    err_ft4 = ft[4] - received_list[9][0]
    print('Feature 4 error: {:.3}'.format(np.max(np.abs(err_ft4))))
    print('Calculation time in clock cycles: {}'.format(received_list[9][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[9][1][0] / 240000))

    err_ft5 = ft[5] - received_list[10][0]
    print('Feature 5 error: {:.3}'.format(np.max(np.abs(err_ft5))))
    print('Calculation time in clock cycles: {}'.format(received_list[10][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[10][1][0] / 240000))

    err_ft6 = ft[6] - received_list[11][0]
    print('Feature 6 error: {:.3}'.format(np.max(np.abs(err_ft6))))
    print('Calculation time in clock cycles: {}'.format(received_list[11][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[11][1][0] / 240000))

    err_ft7 = ft[7] - received_list[12][0]
    print('Feature 7 error: {:.3}'.format(np.max(np.abs(err_ft7))))
    print('Calculation time in clock cycles: {}'.format(received_list[12][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[12][1][0] / 240000))

    err_ft8 = ft[8] - received_list[13][0]
    print('Feature 8 error: {:.3}'.format(np.max(np.abs(err_ft8))))
    print('Calculation time in clock cycles: {}'.format(received_list[13][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[13][1][0] / 240000))

    err_ft9 = ft[9] - received_list[14][0]
    print('Feature 9 error: {:.3}'.format(np.max(np.abs(err_ft9))))
    print('Calculation time in clock cycles: {}'.format(received_list[14][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[14][1][0] / 240000))

    err_ft10 = ft[10] - received_list[15][0]
    print('Feature 10 error: {:.3}'.format(np.max(np.abs(err_ft10))))
    print('Calculation time in clock cycles: {}'.format(received_list[15][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[15][1][0] / 240000))

    err_ft11 = ft[11] - received_list[16][0]
    print('Feature 11 error: {:.3}'.format(np.max(np.abs(err_ft11))))
    print('Calculation time in clock cycles: {}'.format(received_list[16][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[16][1][0] / 240000))

    err_ft12 = ft[12] - received_list[17][0]
    print('Feature 12 error: {:.3}'.format(np.max(np.abs(err_ft12))))
    print('Calculation time in clock cycles: {}'.format(received_list[17][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[17][1][0] / 240000))

    err_ft13 = ft[13] - received_list[18][0]
    print('Feature 13 error: {:.3}'.format(np.max(np.abs(err_ft13))))
    print('Calculation time in clock cycles: {}'.format(received_list[18][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[18][1][0] / 240000))

    err_ft14 = ft[14] - received_list[19][0]
    print('Feature 14 error: {:.3}'.format(np.max(np.abs(err_ft14))))
    print('Calculation time in clock cycles: {}'.format(received_list[19][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[19][1][0] / 240000))

    err_ft15 = ft[15] - received_list[20][0]
    print('Feature 15 error: {:.3}'.format(np.max(np.abs(err_ft15))))
    print('Calculation time in clock cycles: {}'.format(received_list[20][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[20][1][0] / 240000))

    err_ft16 = ft[16] - received_list[21][0]
    print('Feature 15 error: {:.3}'.format(np.max(np.abs(err_ft16))))
    print('Calculation time in clock cycles: {}'.format(received_list[21][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[21][1][0] / 240000))

    err_ft17 = ft[17] - received_list[22][0]
    print('Feature 15 error: {:.3}'.format(np.max(np.abs(err_ft17))))
    print('Calculation time in clock cycles: {}'.format(received_list[22][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[22][1][0] / 240000))

    err_ft18 = ft[18] - received_list[23][0]
    print('Feature 15 error: {:.3}'.format(np.max(np.abs(err_ft18))))
    print('Calculation time in clock cycles: {}'.format(received_list[23][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[23][1][0] / 240000))

    err_ft19 = ft[19] - received_list[24][0]
    print('Feature 15 error: {:.3}'.format(np.max(np.abs(err_ft19))))
    print('Calculation time in clock cycles: {}'.format(received_list[24][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[24][1][0] / 240000))

    err_ft20 = ft[20] - received_list[25][0]
    print('Feature 15 error: {:.3}'.format(np.max(np.abs(err_ft20))))
    print('Calculation time in clock cycles: {}'.format(received_list[25][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[25][1][0] / 240000))

    err_ft21 = ft[21] - received_list[26][0]
    print('Feature 15 error: {:.3}'.format(np.max(np.abs(err_ft21))))
    print('Calculation time in clock cycles: {}'.format(received_list[26][1][0]))
    print('Calculation time in ms: {:.3}'.format(received_list[26][1][0] / 240000))
