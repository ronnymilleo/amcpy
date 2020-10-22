import struct
import time
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import seaborn as sns
import serial
import tensorflow as tf
import wandb
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from wandb.keras import WandbCallback

import features
import functions
from preprocessing import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class HyperParameter:
    def __init__(self, arguments):
        if arguments is None:
            self.default = True
            self.dropout = 0.2
            self.epochs = 25
            self.optimizer = 'adam'
            self.initializer = 'he_normal'
            self.layer_size_hl1 = 22
            self.layer_size_hl2 = 18
            self.layer_size_hl3 = 14
            self.activation = 'sigmoid'
            self.batch_size = 32
        else:
            self.default = False
            self.dropout = round(float(arguments.dropout), 2)
            self.epochs = int(arguments.epochs)
            self.optimizer = arguments.optimizer
            self.activation = arguments.activation
            self.initializer = arguments.initializer
            self.layer_size_hl1 = int(arguments.layer_size_hl1)
            self.layer_size_hl2 = int(arguments.layer_size_hl2)
            self.layer_size_hl3 = int(arguments.layer_size_hl3)
            self.batch_size = 32

    def get_dict(self):
        return dict(dropout=round(float(self.dropout), 2),
                    epochs=int(self.epochs),
                    optimizer=self.optimizer,
                    activation=self.activation,
                    initializer=self.initializer,
                    layer_size_hl1=int(self.layer_size_hl1),
                    layer_size_hl2=int(self.layer_size_hl2),
                    layer_size_hl3=int(self.layer_size_hl3)
                    )


def create_model(cfg: HyperParameter) -> Sequential:  # Return sequential model
    # Here is where the magic really happens! Check this out:
    model = Sequential()  # The model used is the sequential
    # It has a fully connected input layer
    model.add(Dense(number_of_features, activation=cfg.activation,
                    input_shape=(number_of_features,)))
    # With three others hidden layers
    model.add(Dense(cfg.layer_size_hl1, activation=cfg.activation))
    # And a dropout layer between them to avoid overfitting
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl2, activation=cfg.activation))
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl3, activation=cfg.activation))
    model.add(Dense(len(modulation_list), activation='softmax'))
    model.compile(optimizer=cfg.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_from_id(model_id: str):
    if model_id == ' ':  # If you do not specify a RNA id, it'll use the newest available in rna_folder
        aux = [f for f in os.listdir(rna_folder) if "rna" in f]
        rna_files = [join(str(rna_folder), item) for item in aux]
        input_id = max(rna_files, key=os.path.getctime)
        print("\nRNA ID not provided. Using RNA model with id {}, created at {} instead.".format(
            input_id.split("-")[1].split(".")[0], time.ctime(os.path.getmtime(input_id))))
        input_id = input_id.split("-")[1].split(".")[0]
        rna = join(str(rna_folder), "rna-" + input_id + ".h5")
        model = load_model(rna)  # Loads the RNA model
    else:
        rna = join(str(rna_folder), "rna-" + model_id + ".h5")
        model = load_model(rna)
        print("\nUsing RNA with id {}.".format(model_id))
    return model, model_id


def train_rna(cfg: HyperParameter):
    new_id = str(uuid.uuid1()).split('-')[0]  # Generates a unique id to each RNA created

    X_train_tensor = tf.convert_to_tensor(
        X_train, dtype=np.float32, name='X_train'
    )
    X_test_tensor = tf.convert_to_tensor(
        X_test, dtype=np.float32, name='X_test'
    )
    # Create encoded labels using tensorflow's one-hot function (to use categorical_crossentropy)
    depth = len(modulation_list)
    y_train_tensor = tf.one_hot(y_train, depth)
    y_test_tensor = tf.one_hot(y_test, depth)

    model = create_model(cfg)
    # Once created, the model is then compiled, trained and saved for further evaluation
    # batch_size = number of samples per gradient update
    # verbose = 0 = silent, 1 = progress bar, 2 = one line per epoch
    if cfg.default:
        # Default mode
        history = model.fit(X_train_tensor, y_train_tensor, batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2)
    else:
        # Execute code only if wandb is activated
        history = model.fit(X_train_tensor, y_train_tensor, batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2,
                            callbacks=[WandbCallback(validation_data=(X_test_tensor, y_test_tensor))])

    model.save(str(join(rna_folder, 'rna-' + new_id + '.h5')))
    model.save_weights(str(join(rna_folder, 'weights-' + new_id + '.h5')))
    print(join("\nRNA saved with id ", new_id, "\n").replace("\\", ""))

    # Here is where we make the first evaluation of the RNA
    loss, acc = model.evaluate(X_test_tensor, y_test_tensor, batch_size=cfg.batch_size, verbose=2)
    print('Test Accuracy: %.3f' % acc)

    # Execute code only if wandb is activated
    if cfg.default is False:
        # Here, WANDB takes place and logs all metrics to the cloud
        metrics = {'accuracy': acc,
                   'loss': loss,
                   'dropout': cfg.dropout,
                   'epochs': cfg.epochs,
                   'initializer': cfg.initializer,
                   'layer_syze_hl1': cfg.layer_size_hl1,
                   'layer_syze_hl2': cfg.layer_size_hl2,
                   'layer_syze_hl3': cfg.layer_size_hl3,
                   'optimizer': cfg.optimizer,
                   'activation': cfg.activation,
                   'id': new_id}
        wandb.log(metrics)

    # Here we make a prediction using the test data...
    print('\nStarting prediction')
    predict = np.argmax(model.predict(X_test_tensor, batch_size=cfg.batch_size), axis=-1)

    # And create a Confusion Matrix for a better visualization!
    print('\nConfusion Matrix:')
    confusion_matrix = tf.math.confusion_matrix(tf.argmax(y_test_tensor, axis=1), predict).numpy()
    confusion_matrix_normalized = np.around(
        confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
        decimals=2)
    print(confusion_matrix_normalized)
    cm_data_frame = pd.DataFrame(confusion_matrix_normalized, index=modulation_list, columns=modulation_list)
    figure = plt.figure(figsize=(8, 4), dpi=150)
    sns.heatmap(cm_data_frame, annot=True, cmap=plt.cm.get_cmap('Blues', 6))
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(fig_folder, 'confusion_matrix-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_accuracy-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_loss-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.close(figure)


def evaluate_rna(evaluate_loaded_model):  # Make a prediction using some samples
    print("\nStarting RNA evaluation by SNR.")
    # For each modulation, randomly loads the test_size samples
    # and predict the result to all SNR values
    result = np.zeros((len(modulation_list), number_of_snr))
    for i, mod in enumerate(features_files):
        print("Evaluating {}".format(mod.split("_")[0]))
        with open(join(data_folder, mod), 'rb') as evaluating_data:
            data = pickle.load(evaluating_data)
        for j, snr in enumerate(snr_list):
            X_dataset = data[j, :]  # Test over all available data
            # Fit into data used for training, results are means and variances used to standardise the data
            X_dataset = scaler.transform(X_dataset)
            right_label = [info_json['modulations']['labels'][mod.split("_")[0]] for _ in range(len(X_dataset))]
            predict = np.argmax(evaluate_loaded_model.predict(X_dataset), axis=-1)
            accuracy = accuracy_score(right_label, predict)
            result[i][j] = accuracy

    accuracy_graphic(result)

    if not os.path.isfile(join(rna_folder, "weights-" + loaded_model_id + ".h5")):
        print("Weights file not found. Saving it into RNA folder")
        evaluate_loaded_model.save_weights(str(join(rna_folder, 'weights-' + loaded_model_id + '.h5')))


def accuracy_graphic(result):
    # Then, it creates an accuracy graphic, containing the
    # prediction result to all snr values and all modulations
    # figure = plt.figure(figsize=(8, 4), dpi=150)
    plt.title("Accuracy")
    plt.ylabel("Right prediction")
    plt.xlabel("SNR [dB]")
    plt.xticks(np.arange(number_of_snr), [info_json['snr']['values'][i] for i in info_json['snr']['using']])
    for item in range(len(result)):
        plt.plot(result[item], label=modulation_list[item])
    plt.legend(loc='best')
    plt.savefig(join(fig_folder, "accuracy-" + loaded_model_id + ".png"),
                bbox_inches='tight', dpi=300)
    plt.show()
    # figure.clf()
    # plt.close(figure)


def quantize_data(input_array, q_type: tf.dtypes):
    min_range = np.min(input_array)
    # print('Min value: {}'.format(min_range))
    max_range = np.max(input_array)
    # print('Max value: {}'.format(max_range))
    quantized_data = tf.quantization.quantize(
        input_array, min_range, max_range, q_type, mode='SCALED'
    )
    return quantized_data


def quantize_rna(input_array, q_type: tf.dtypes):
    min_range_w = np.min(input_array[0])
    min_range_b = np.min(input_array[1])
    # Debug
    # print('Min value for weights: {}'.format(min_range_w))
    # print('Min value for bias: {}'.format(min_range_b))
    max_range_w = np.max(input_array[0])
    max_range_b = np.max(input_array[1])
    # Debug
    # print('Max value for weights: {}'.format(max_range_w))
    # print('Max value for bias: {}'.format(max_range_b))
    quantized_weights = tf.quantization.quantize(
        input_array[0], min_range_w, max_range_w, q_type, mode='SCALED',
        round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None,
        ensure_minimum_range=0.01
    )
    quantized_bias = tf.quantization.quantize(
        input_array[1], min_range_b, max_range_b, q_type, mode='SCALED',
        round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None,
        ensure_minimum_range=0.01
    )
    return list([quantized_weights, quantized_bias])


def dequantize_rna(original_array, input_array):
    min_range_w = np.min(original_array[0])
    min_range_b = np.min(original_array[1])
    # Debug
    # print('Min value for weights: {}'.format(min_range_w))
    # print('Min value for bias: {}'.format(min_range_b))
    max_range_w = np.max(original_array[0])
    max_range_b = np.max(original_array[1])
    # Debug
    # print('Max value for weights: {}'.format(max_range_w))
    # print('Max value for bias: {}'.format(max_range_b))
    dequantized_weights = tf.quantization.dequantize(
        input_array[0].output, min_range_w, max_range_w, mode='SCALED', name=None, axis=None,
        narrow_range=False, dtype=tf.dtypes.float32
    )
    dequantized_bias = tf.quantization.dequantize(
        input_array[1].output, min_range_b, max_range_b, mode='SCALED', name=None, axis=None,
        narrow_range=False, dtype=tf.dtypes.float32
    )
    return list([dequantized_weights, dequantized_bias])


def get_quantization_error(original_weights, dequantized_weights):
    # TODO: quantization error graphics
    err = original_weights - dequantized_weights
    # plt.plot(err)
    # plt.show()
    return err


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


def serial_communication():
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

    # Write to UART
    print('Transmitting Neural Network...')
    for point in range(0, 1700):
        binary = struct.pack('<h', weights[point])
        ser.write(binary)
    for point in range(0, 82):
        binary = struct.pack('<h', biases[point])
        ser.write(binary)

    received_wandb = receive_wandb(ser, 1782)

    err_weights = received_wandb[0:1700] - weights
    if np.max(err_weights) == 0:
        print('Weights loaded successfully')
    else:
        print('Error loading weights')
    err_biases = received_wandb[1700:1782] - biases
    if np.max(err_biases) == 0:
        print('Biases loaded successfully')
    else:
        print('Error loading biases')

    print('Wait...')
    time.sleep(3)
    print('Starting modulation signals transmission!')

    for mod in modulation_list:
        # Create error information arrays and features results from microcontroller
        err_abs_vector = []
        err_phase_vector = []
        err_unwrapped_phase_vector = []
        err_freq_vector = []
        err_cn_abs_vector = []
        err_features = []
        predictions = []
        gathered_data = []
        parsed_signal = data_mat[info[mod]]
        print('Modulation = ' + mod)
        for snr in snr_range:
            print('SNR = {}'.format(snr))
            for frame in frame_range:
                print('Frame = {}'.format(frame))
                i_values = functions.InstValues(parsed_signal[snr, frame, 0:frameSize])
                ft = np.float32(features.calculate_features(parsed_signal[snr, frame, 0:frameSize]))
                q_ft = quantize_data(ft, q_type=tf.dtypes.qint16)
                print('Quantized features: {}'.format(q_ft))
                # Write to UART
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
                # print('Inst absolute max error: {:.3}'.format(np.max(np.abs(err_abs_vector))))
                # print('Calc time in clock cycles: {}'.format(received_list[1][1][0]))
                # print('Calc time in ms: {:.3}'.format(received_list[1][1][0] / 200000))
                #
                # err_phase_vector.append(i_values.inst_phase.T - received_list[2][0])
                # print('Inst phase max error: {:.3}'.format(np.max(np.abs(err_phase_vector))))
                # print('Calc time in clock cycles: {}'.format(received_list[2][1][0]))
                # print('Calc time in ms: {:.3}'.format(received_list[2][1][0] / 200000))
                #
                # err_unwrapped_phase_vector.append(i_values.inst_unwrapped_phase.T - received_list[3][0])
                # print('Inst unwrapped phase max error: {:.3}'.format(np.max(np.abs(err_unwrapped_phase_vector))))
                # print('Calc time in clock cycles: {}'.format(received_list[3][1][0]))
                # print('Calc time in ms: {:.3}'.format(received_list[3][1][0] / 200000))
                #
                # err_freq_vector.append(i_values.inst_freq[0:frameSize - 1].T - received_list[4][0][0:frameSize - 1])
                # print('Inst frequency max error: {:.3}'.format(np.max(np.abs(err_freq_vector))))
                # print('Calc time in clock cycles: {}'.format(received_list[4][1][0]))
                # print('Calc time in ms: {:.3}'.format(received_list[4][1][0] / 200000))
                #
                # err_cn_abs_vector.append(i_values.inst_cna.T - received_list[5][0])
                # print('Inst CN amplitude max error: {:.3}'.format(np.max(np.abs(err_cn_abs_vector))))
                # print('Calc time in clock cycles: {}'.format(received_list[5][1][0]))
                # print('Calc time in ms: {:.3}'.format(received_list[5][1][0] / 200000))

                # err_features.append(ft - received_list[6][0])
                # print('Error list: {}'.format(err_features))
                # print('Timings list: {}'.format(received_list[6][1]))
                # print('Timings list (ms): {}'.format(received_list[6][1] / 200000))

                err_features.append(ft - received_list[0][0])
                print('Error list: {}'.format(err_features))
                print('Timings list: {}'.format(received_list[0][1]))
                print('Timings list (ms): {}'.format(received_list[0][1] / 200000))

                predictions.append(received_list[1][0])
                print('Predictions for ' + mod + ': {}'.format(predictions))
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
    with open("./info.json") as handle:
        info_json = json.load(handle)

    frameSize = info_json['frameSize']

    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))
    # parser = argparse.ArgumentParser(description='RNA argument parser')
    # parser.add_argument('--dropout', action='store', dest='dropout')
    # parser.add_argument('--epochs', action='store', dest='epochs')
    # parser.add_argument('--optimizer', action='store', dest='optimizer')
    # parser.add_argument('--initializer', action='store', dest='initializer')
    # parser.add_argument('--layer_size_hl1', action='store', dest='layer_size_hl1')
    # parser.add_argument('--layer_size_hl2', action='store', dest='layer_size_hl2')
    # parser.add_argument('--layer_size_hl3', action='store', dest='layer_size_hl3')
    # parser.add_argument('--activation', action='store', dest='activation')
    # arguments = parser.parse_args()
    # wandb.init(project="amcpy-team", config=HyperParameter(arguments).get_dict())
    # config = wandb.config
    X_train, X_test, y_train, y_test, scaler = preprocess_data()

    # Find min and max values for quantization
    global_min = np.min([np.min(X_train), np.min(X_test)])
    global_max = np.max([np.max(X_train), np.max(X_test)])
    print('Global min and max:')
    print('Min: {}'.format(global_min))
    print('Max: {}'.format(global_max))

    # train_rna(HyperParameter(None))
    loaded_model, loaded_model_id = get_model_from_id(' ')
    evaluate_rna(loaded_model)

    layer_numbers = [0, 1, 3, 5, 6]
    quantized = []
    dequantized_w = []
    k = 0
    for ln in layer_numbers:
        quantized.append(quantize_rna(loaded_model.layers[ln].get_weights(), q_type=tf.dtypes.qint16))
        dequantized_w.append(dequantize_rna(loaded_model.layers[ln].get_weights(), quantized[k]))
        error_w = get_quantization_error(loaded_model.layers[ln].get_weights()[0], dequantized_w[k][0])
        print('Max error INPUT LAYER {} W: {}'.format(ln, np.max(error_w)))
        error_b = get_quantization_error(loaded_model.layers[ln].get_weights()[1], dequantized_w[k][1])
        print('Max error INPUT LAYER {} B: {}'.format(ln, np.max(error_b)))
        k = k + 1

    # Convert quantized weights into numpy arrays
    l1 = np.reshape(quantized[0][0][0].numpy(), (22 * 22,))
    b1 = quantized[0][1][0].numpy()
    l2 = np.reshape(quantized[1][0][0].numpy(), (22 * 22,))
    b2 = quantized[1][1][0].numpy()
    l3 = np.reshape(quantized[2][0][0].numpy(), (22 * 18,))
    b3 = quantized[2][1][0].numpy()
    l4 = np.reshape(quantized[3][0][0].numpy(), (18 * 14,))
    b4 = quantized[3][1][0].numpy()
    l5 = np.reshape(quantized[4][0][0].numpy(), (14 * 6,))
    b5 = quantized[4][1][0].numpy()

    weights = np.concatenate((l1, l2, l3, l4, l5))
    biases = np.concatenate((b1, b2, b3, b4, b5))

    serial_communication()
