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


def receive_data(port: serial.Serial(), frameSize) -> (float, int):
    # print('Receiving data...')
    data = []
    real = np.zeros((2048,), dtype=np.float32)
    imag = np.zeros((2048,), dtype=np.float32)
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
        num_array = np.zeros((22,), dtype=np.float32)
        counter_array = np.zeros((22,), dtype=np.int32)
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
    return num_array, counter_array, real, imag


def serial_communication():
    with open("./info.json") as handle:
        info_json = json.load(handle)

    modulations = info_json['modulations']['names']
    frameSize = info_json['frameSize']

    # Filename setup
    mat_file_name = pathlib.Path(join(os.getcwd(), 'mat-data', 'all_modulations_data.mat'))

    # Dictionary to access variable inside MAT file
    info = {'BPSK': 'signal_bpsk',
            'QPSK': 'signal_qpsk',
            'PSK8': 'signal_8psk',
            'QAM16': 'signal_qam16',
            'QAM64': 'signal_qam64',
            'noise': 'signal_noise'}

    # Load MAT file and parse data
    data_mat = scipy.io.loadmat(mat_file_name)
    print(str(mat_file_name) + ' file loaded...')

    # Setup UART COM on Windows
    ser = serial.Serial(port='COM3', baudrate=115200, parity='N', bytesize=8, stopbits=1, timeout=1)

    snr_range = np.linspace(11, 15, 5, dtype=np.int16)  # 12 14 16 18 20
    frame_range = np.linspace(0, 4, 5, dtype=np.int16)

    # # Write to UART
    # print('Transmitting Neural Network...')
    # for point in range(0, 1700):
    #     binary = struct.pack('<h', weights[point])
    #     ser.write(binary)
    # for point in range(0, 82):
    #     binary = struct.pack('<h', biases[point])
    #     ser.write(binary)
    #
    # print('Wait...')
    # time.sleep(3)

    for mod in modulations:
        # Create error information arrays and features results from microcontroller
        err_abs_vector = []
        err_phase_vector = []
        err_unwrapped_phase_vector = []
        err_freq_vector = []
        err_cn_abs_vector = []
        err_features = []
        gathered_data = []
        parsed_signal = data_mat[info[mod]]
        print('Modulation = ' + mod)
        for snr in snr_range:
            print('SNR = {}'.format(snr))
            for frame in frame_range:
                print('Frame = {}'.format(frame))
                i_values = np.float32(functions.InstValues(parsed_signal[snr, frame, 0:frameSize]))
                ft = np.float32(features.calculate_features(parsed_signal[snr, frame, 0:frameSize]))
                # Write to UART
                print('Transmitting...')
                for point in range(0, 2048):
                    binary = struct.pack('<f', np.real(parsed_signal[snr, frame, point]))
                    ser.write(binary)
                    binary = struct.pack('<f', np.imag(parsed_signal[snr, frame, point]))
                    ser.write(binary)

                received_list = []
                for results in range(0, 8):
                    num_array, counter_array, real, imag = receive_data(ser, frameSize)
                    if results == 0:
                        err = False
                        for n in range(0, 2048):
                            if abs(real[n] - np.real(parsed_signal[snr, frame, n])) > 0:
                                err = True
                                print('Error at real sample number {}'.format(n))
                            if abs(imag[n] - np.imag(parsed_signal[snr, frame, n])) > 0:
                                err = True
                                print('Error at real sample number {}'.format(n))
                        if err:
                            received_list.append((0, 0))
                        else:
                            print('Echo ok - data validated')
                            received_list.append([real, imag])
                    else:
                        received_list.append([num_array, counter_array])

                err_abs_vector.append(i_values.inst_abs.T - received_list[1][0])
                print('Instantaneous absolute max error: {:.3}'.format(np.max(np.abs(err_abs_vector))))
                print('Calculation time in clock cycles: {}'.format(received_list[1][1][0]))
                print('Calculation time in ms: {:.3}'.format(received_list[1][1][0] / 200000))

                err_phase_vector.append(i_values.inst_phase.T - received_list[2][0])
                print('Instantaneous phase max error: {:.3}'.format(np.max(np.abs(err_phase_vector))))
                print('Calculation time in clock cycles: {}'.format(received_list[2][1][0]))
                print('Calculation time in ms: {:.3}'.format(received_list[2][1][0] / 200000))

                err_unwrapped_phase_vector.append(i_values.inst_unwrapped_phase.T - received_list[3][0])
                print(
                    'Instantaneous unwrapped phase max error: {:.3}'.format(np.max(np.abs(err_unwrapped_phase_vector))))
                print('Calculation time in clock cycles: {}'.format(received_list[3][1][0]))
                print('Calculation time in ms: {:.3}'.format(received_list[3][1][0] / 200000))

                err_freq_vector.append(i_values.inst_freq[0:frameSize - 1].T - received_list[4][0][0:frameSize - 1])
                print('Instantaneous frequency max error: {:.3}'.format(np.max(np.abs(err_freq_vector))))
                print('Calculation time in clock cycles: {}'.format(received_list[4][1][0]))
                print('Calculation time in ms: {:.3}'.format(received_list[4][1][0] / 200000))

                err_cn_abs_vector.append(i_values.inst_cna.T - received_list[5][0])
                print('Instantaneous CN amplitude max error: {:.3}'.format(np.max(np.abs(err_cn_abs_vector))))
                print('Calculation time in clock cycles: {}'.format(received_list[5][1][0]))
                print('Calculation time in ms: {:.3}'.format(received_list[5][1][0] / 200000))

                err_features.append(ft - received_list[6][0])
                print('Error list: {}'.format(err_features))
                print('Timings list: {}'.format(received_list[6][1]))
                print('Timings list (ms): {}'.format(received_list[6][1] / 200000))

                print('Wait...')
                gathered_data.append(received_list)
                save_dict = {'Modulation': mod, 'SNR': snr, 'Frame': frame,
                             'Data': received_list, 'inst_values': i_values,
                             'features': ft}
                file_name = mod + '_' + str(snr) + '_' + str(frame) + '.mat'
                scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'arm-data', file_name)), save_dict)
                time.sleep(3)

        save_dict = {'Data': gathered_data, 'err_abs_vector': err_abs_vector, 'err_phase_vector': err_phase_vector,
                     'err_unwrapped_phase_vector': err_unwrapped_phase_vector, 'err_freq_vector': err_freq_vector,
                     'err_cn_abs_vector': err_cn_abs_vector, 'err_features': err_features, 'snr_range': snr_range,
                     'frame_range': frame_range, 'modulation': mod}
        scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'arm-data', mod + '.mat')), save_dict)


if __name__ == '__main__':
    serial_communication()
