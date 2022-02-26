import struct
import time

import scipy.io
import serial

import functions
import quantization
from globals import *


def receive_data(port: serial.Serial()) -> (float, int):
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
    if len(data) == len(used_features) * 2:
        num_array = np.zeros((len(used_features),), dtype=np.float32)
        counter_array = np.zeros((len(used_features),), dtype=np.int32)
    else:
        num_array = np.zeros((len(data) - 1,), dtype=np.float32)
        counter_array = np.zeros((1,), dtype=np.int32)
    if len(data) == 2:  # Prediction
        print('Prediction')
        new_data = b''.join(data)
        num_array = struct.unpack('<f', new_data[0:4])
        counter_array = struct.unpack('<i', new_data[4:8])
    elif len(data) == len(used_features) * 2:  # Features and counters
        print('Features')
        new_data = b''.join(data)
        for x in range(0, len(used_features)):
            aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
            num_array[x] = aux[0]
        for x in range(len(used_features), len(used_features) * 2):
            aux = struct.unpack('<i', new_data[x * 4:x * 4 + 4])
            counter_array[x - len(used_features)] = aux[0]
    elif len(data) == frame_size + 1:  # Data
        print('Inst values')
        new_data = b''.join(data)
        for x in range(0, frame_size):
            aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
            num_array[x] = aux[0]
        counter_array = struct.unpack('<i', new_data[frame_size * 4:frame_size * 4 + 4])
    elif len(data) > frame_size + 1:  # Echo
        print('Echo')
        new_data = b''.join(data)
        for x in range(0, frame_size * 8, 8):
            r = struct.unpack('<f', new_data[x:x + 4])
            i = struct.unpack('<f', new_data[x + 4:x + 8])
            real[x // 8] = r[0]
            imag[x // 8] = i[0]
        num_array = None
        counter_array = None
    return num_array, counter_array, real, imag


def receive_wandb(port, size):
    # print('Receiving data...')
    data = []
    np_array = np.zeros((size,), dtype=np.int16)
    start = 0
    while True:
        a = port.read(size=2)
        if a == b'\xCA\xCA':
            # print('Head')
            start = 1
        elif a == b'\xF0\xF0':
            # print('Tail')
            break
        elif start == 1:
            data.append(a)

    for x in range(0, size):
        aux = struct.unpack('<h', data[x])
        np_array[x] = aux[0]

    return np_array


def receive_scaler(port, size):
    # print('Receiving data...')
    data = []
    np_array = np.zeros((size,), dtype=np.float32)
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

    new_data = b''.join(data)
    for x in range(0, len(used_features)):
        aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
        np_array[x] = aux[0]
    for x in range(len(used_features), len(used_features) * 2):
        aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
        np_array[x] = aux[0]

    return np_array


def serial_communication(weights, biases, scaler, info_dict):
    # Setup UART COM on Windows
    ser = serial.Serial(port='COM3', baudrate=115200, parity='N', bytesize=8, stopbits=1, timeout=1)

    snr_range = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # snr_range = np.linspace(15, 0, 16, dtype=np.int16)  # 10 12 14 16 18 20
    frame_range = np.linspace(0, 99, 100, dtype=np.int16)

    # Write to UART
    print('Transmitting Neural Network...')
    for point in range(0, len(weights)):
        binary = struct.pack('<h', weights[point])
        ser.write(binary)
    for point in range(0, len(biases)):
        binary = struct.pack('<h', biases[point])
        ser.write(binary)

    received_wandb = receive_wandb(ser, len(weights) + len(biases))
    err_weights = received_wandb[0:len(weights)] - weights
    if np.max(err_weights) == 0:
        print('Weights loaded successfully')
        print(weights)
    else:
        print('Error loading weights')
    err_biases = received_wandb[len(weights):len(weights) + len(biases)] - biases
    if np.max(err_biases) == 0:
        print('Biases loaded successfully')
        print(biases)
    else:
        print('Error loading biases')

    print('Wait...')
    time.sleep(3)

    print('Transmit scaler for Standardization')
    # Write to UART
    for point in range(0, len(used_features)):
        binary = struct.pack('<f', np.float32(scaler.mean_[point]))
        ser.write(binary)
    for point in range(0, len(used_features)):
        binary = struct.pack('<f', np.float32(scaler.scale_[point]))
        ser.write(binary)

    received_scaler = receive_scaler(ser, len(used_features) * 2)
    err_scaler = received_scaler - np.concatenate((np.float32(scaler.mean_), np.float32(scaler.scale_)))
    if np.max(err_scaler) == 0:
        print('Scaler loaded successfully')
        print(received_scaler)
    else:
        print('Error loading scaler')

    print('Wait...')
    time.sleep(3)

    print('Starting modulation signals transmission!')
    for mod in modulation_signals_with_noise:
        # Create error information arrays and features results from microcontroller
        err_abs_vector = []
        err_phase_vector = []
        err_unwrapped_phase_vector = []
        err_freq_vector = []
        err_cn_abs_vector = []
        err_features = []
        predictions = []
        gathered_data = []

        # Filename setup
        mat_file_name = pathlib.Path(join(os.getcwd(), 'mat-data', 'all_modulations.mat'))

        # Load MAT file and parse data
        data_mat = scipy.io.loadmat(mat_file_name.__str__())

        print(str(mat_file_name) + ' file loaded...')
        parsed_signal = data_mat[mat_info[mod]]
        for snr in snr_range:
            for frame in frame_range:
                print('Modulation = ' + mod)
                print('SNR = {}'.format(snr))
                print('Frame = {}'.format(frame))
                i_values = functions.InstValues(parsed_signal[snr, frame, 0:frame_size])
                ft = np.float32(functions.calculate_features(used_features, parsed_signal[snr, frame, 0:frame_size]))
                ft_scaled = (ft - np.float32(scaler.mean_)) / np.float32(scaler.scale_)
                q_format = info_dict["Input"]
                q_ft = quantization.quantize_data(ft_scaled, q_format)
                print('Features: {}'.format(ft_scaled))
                print('Features Q Format: ' + q_format)
                print('Quantized features: {}'.format(q_ft))
                print('Transmitting...')
                for point in range(0, 2048):
                    binary = struct.pack('<f', np.real(parsed_signal[snr, frame, point]))
                    ser.write(binary)
                    binary = struct.pack('<f', np.imag(parsed_signal[snr, frame, point]))
                    ser.write(binary)

                received_list = []
                for results in range(1, 3):
                    num_array, counter_array, real, imag = receive_data(ser)
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

                # err_abs_vector.append(i_values.inst_abs.T - received_list[1][0])
                # print('Err abs: {}'.format(np.max(err_abs_vector)))
                # err_phase_vector.append(i_values.inst_phase.T - received_list[2][0])
                # print('Err phase: {}'.format(np.max(err_phase_vector)))
                # err_unwrapped_phase_vector.append(i_values.inst_unwrapped_phase.T - received_list[3][0])
                # print('Err unwrapped phase: {}'.format(np.max(err_unwrapped_phase_vector)))
                # err_freq_vector.append(i_values.inst_freq[0:frame_size - 1].T - received_list[4][0][0:frame_size - 1])
                # print('Err freq: {}'.format(np.max(err_freq_vector)))
                # err_cn_abs_vector.append(i_values.inst_cna.T - received_list[5][0])
                # print('Err CN abs: {}'.format(np.max(err_cn_abs_vector)))
                err_features.append(ft - received_list[0][0])
                print('Err features: {}'.format(np.max(err_features)))

                predictions.append(received_list[1][0])
                print(predictions)
                correct = 0
                for p in predictions:
                    if mod == 'BPSK' and p == (0.0,):
                        correct += 1
                    elif mod == 'QPSK' and p == (1.0,):
                        correct += 1
                    elif mod == '8PSK' and p == (2.0,):
                        correct += 1
                    elif mod == '16QAM' and p == (3.0,):
                        correct += 1
                    elif mod == '64QAM' and p == (4.0,):
                        correct += 1
                    elif mod == 'noise' and p == (5.0,):
                        correct += 1
                    else:
                        correct += 0

                print('Last prediction = {}'.format(received_list[1][0]))
                print('Accuracy = {}'.format(correct * 100 / len(predictions)))

                print('Wait...')
                gathered_data.append(received_list)
                time.sleep(2.5)

        save_dict = {'Data': gathered_data, 'err_abs_vector': err_abs_vector, 'err_phase_vector': err_phase_vector,
                     'err_unwrapped_phase_vector': err_unwrapped_phase_vector, 'err_freq_vector': err_freq_vector,
                     'err_cn_abs_vector': err_cn_abs_vector, 'err_features': err_features, 'snr_range': snr_range,
                     'frame_range': frame_range, 'modulation': mod, 'predictions': predictions}
        scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'arm-data', mod + '.mat')), save_dict)
