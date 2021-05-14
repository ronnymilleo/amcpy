import argparse
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
from keras.optimizers import RMSprop, Adam, Nadam
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from wandb.keras import WandbCallback

from old import features
import functions
import quantization
from preprocessing import *

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, enable=True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


class HyperParameter:
    def __init__(self, arguments):
        if arguments is None:
            self.default = True
            self.activation = 'relu'
            self.batch_size = 32
            self.dropout = 0.4
            self.epochs = 10
            self.initializer = 'he_normal'
            self.layer_size_hl1 = 18
            self.layer_size_hl2 = 18
            self.layer_size_hl3 = 18
            self.learning_rate = 1e-3
            self.optimizer = 'rmsprop'
        else:
            self.default = False
            self.activation = arguments.activation
            self.batch_size = int(arguments.batch_size)
            self.dropout = round(float(arguments.dropout), 2)
            self.epochs = int(arguments.epochs)
            self.initializer = 'he_normal'
            self.layer_size_hl1 = int(arguments.layer_size_hl1)
            self.layer_size_hl2 = int(arguments.layer_size_hl2)
            self.layer_size_hl3 = int(arguments.layer_size_hl3)
            self.learning_rate = float(arguments.learning_rate)
            self.optimizer = arguments.optimizer

    def get_dict(self):
        return dict(activation=self.activation,
                    batch_size=int(self.batch_size),
                    dropout=round(float(self.dropout), 2),
                    epochs=int(self.epochs),
                    initializer=self.initializer,
                    layer_size_hl1=int(self.layer_size_hl1),
                    layer_size_hl2=int(self.layer_size_hl2),
                    layer_size_hl3=int(self.layer_size_hl3),
                    learning_rate=self.learning_rate,
                    optimizer=self.optimizer
                    )


def create_model(cfg: HyperParameter) -> Sequential:  # Return sequential model
    model = Sequential()
    model.add(Dense(number_of_used_features,
                    activation=cfg.activation,
                    input_shape=(number_of_used_features,),
                    kernel_regularizer=regularizers.l2(0.001)))

    # model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl1,
                    activation=cfg.activation,
                    kernel_regularizer=regularizers.l2(0.001)))

    # model.add(Dropout(cfg.dropout))
    # model.add(Dense(cfg.layer_size_hl2,
    #                 activation=cfg.activation,
    #                 kernel_regularizer=regularizers.l2(0.001)))
    #
    # model.add(Dropout(cfg.dropout))
    # model.add(Dense(cfg.layer_size_hl3,
    #                 activation=cfg.activation,
    #                 kernel_regularizer=regularizers.l2(0.001)))

    # model.add(Dropout(cfg.dropout))
    model.add(Dense(len(signals),
                    activation='softmax'))

    if cfg.optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == 'adam':
        optimizer = Adam(learning_rate=cfg.learning_rate)
    else:
        optimizer = Nadam(learning_rate=cfg.learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


def get_model_from_id(model_id_str: str):
    if model_id_str == ' ':  # If you do not specify a RNA id, it'll use the newest available in rna_folder
        aux = [f for f in os.listdir(rna_folder) if "rna" in f]
        rna_files = [join(str(rna_folder), item) for item in aux]
        input_id = max(rna_files, key=os.path.getctime)
        print("\nRNA ID not provided. Using RNA model with id {}, created at {} instead.".format(
            input_id.split("-")[1].split(".")[0], time.ctime(os.path.getmtime(input_id))))
        input_id = input_id.split("-")[1].split(".")[0]
        rna = join(str(rna_folder), "rna-" + input_id + ".h5")
        model = load_model(rna)  # Loads the RNA model
    else:
        rna = join(str(rna_folder), "rna-" + model_id_str + ".h5")
        model = load_model(rna)
        print("\nUsing RNA with id {}.".format(model_id_str))
    return model, model_id_str


def train_rna(cfg):
    new_id = str(uuid.uuid1()).split('-')[0]  # Generates a unique id to each RNA created
    # Create tensors from dataset
    X_train_tensor = tf.convert_to_tensor(
        X_train, dtype=np.float32, name='X_train'
    )
    X_test_tensor = tf.convert_to_tensor(
        X_test, dtype=np.float32, name='X_test'
    )
    # Create encoded labels using tensorflow's one-hot function (to use categorical_crossentropy)
    depth = len(signals)
    y_train_tensor = tf.one_hot(y_train, depth)
    y_test_tensor = tf.one_hot(y_test, depth)
    # Instantiate model
    model = create_model(cfg)
    # Once created, the model is then compiled, trained and saved for further evaluation
    # batch_size = number of samples per gradient update
    # verbose >> 0 = silent, 1 = progress bar, 2 = one line per epoch

    # Execute code only if wandb is activated
    history = model.fit(X_train_tensor, y_train_tensor, batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2,
                        callbacks=[WandbCallback(validation_data=(X_test_tensor, y_test_tensor))],
                        validation_data=(X_test_tensor, y_test_tensor), use_multiprocessing=True, workers=12)

    model.save(str(join(rna_folder, 'rna-' + new_id + '.h5')))
    model.save_weights(str(join(rna_folder, 'weights-' + new_id + '.h5')))
    print(join("\nRNA saved with id ", new_id, "\n").replace("\\", ""))

    # Here is where we make the first evaluation of the RNA
    loss, acc = model.evaluate(X_train_tensor, y_train_tensor,
                               batch_size=cfg.batch_size,
                               verbose=2)
    print('Test Accuracy: {}'.format(acc))

    # Here, WANDB takes place and logs all metrics to the cloud
    metrics = {'accuracy': acc,
               'loss': loss,
               'activation': cfg.activation,
               'batch_size': cfg.batch_size,
               'dropout': cfg.dropout,
               'epochs': cfg.epochs,
               'initializer': cfg.initializer,
               'layer_size_hl1': cfg.layer_size_hl1,
               'layer_size_hl2': cfg.layer_size_hl2,
               'layer_size_hl3': cfg.layer_size_hl3,
               'learning_rate': cfg.learning_rate,
               'optimizer': cfg.optimizer,
               'id': new_id}
    wandb.log(metrics)

    # Here we make a prediction using the test data...
    print('\nStarting prediction')
    predict = np.argmax(model.predict(X_test), axis=-1)

    # And create a Confusion Matrix for a better visualization!
    print('\nConfusion Matrix:')
    confusion_matrix = tf.math.confusion_matrix(y_test, predict).numpy()
    confusion_matrix_normalized = np.around(
        confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
        decimals=2)
    print(confusion_matrix_normalized)
    cm_data_frame = pd.DataFrame(confusion_matrix_normalized, index=signals, columns=signals)
    figure = plt.figure(figsize=(8, 4), dpi=150)
    sns.heatmap(cm_data_frame, annot=True, cmap=plt.cm.get_cmap('Blues', 6))
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(fig_folder, 'confusion_matrix-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_accuracy-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_loss-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.close(figure)
    return new_id


def evaluate_rna(evaluate_loaded_model):  # Make a prediction using some samples
    print("\nStarting RNA evaluation by SNR.")

    # For each modulation, randomly loads the test_size samples
    # and predict the result to all SNR values
    result = np.zeros((len(signals), len(testing_SNR)))
    for i, mod in enumerate(features_files):
        print("Evaluating {}".format(mod.split("_")[0]))
        data_dict = scipy.io.loadmat(join(data_folder, mod))
        data = data_dict[mat_info[mod.split("_")[0]]]
        for SNR in testing_SNR:
            X_dataset = data[SNR, :, :]  # Test over all available data
            # Fit into data used for training, results are means and variances used to standardise the data
            X_dataset = scaler.transform(X_dataset)
            right_label = [signals_labels[i] for _ in range(len(X_dataset))]
            predict = np.argmax(evaluate_loaded_model.predict(X_dataset), axis=-1)
            accuracy = accuracy_score(right_label, predict)
            result[i][SNR] = accuracy

    accuracy_graphic(result)
    save_dict = {'acc': result}
    scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'figures', loaded_model_id + '_figure_data' + '.mat')), save_dict)

    if not os.path.isfile(join(rna_folder, "weights-" + loaded_model_id + ".h5")):
        print("Weights file not found. Saving it into RNA folder")
        evaluate_loaded_model.save_weights(str(join(rna_folder, 'weights-' + loaded_model_id + '.h5')))


def evaluate_ann():
    n = number_of_testing_frames
    result = np.zeros((len(signals), len(testing_SNR)))
    for SNR, _ in enumerate(testing_SNR):
        for i in range(0, len(signals)):
            predict = np.argmax(loaded_model.predict(X_test_2[n * (i + SNR * len(signals)): n * (i + 1 + SNR * len(signals))]), axis=-1)
            accuracy = accuracy_score(y_test_2[n * (i + SNR * len(signals)): n * (i + 1 + SNR * len(signals))], predict)
            result[i][SNR] = accuracy
    accuracy_graphic(result)
    return result


def accuracy_graphic(result):
    # Then, it creates an accuracy graphic, containing the
    # prediction result to all snr values and all modulations
    plt.figure(figsize=(6, 3), dpi=300)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("SNR [dB]")
    plt.xticks(np.arange(len(testing_SNR)), [SNR_values[i] for i in testing_SNR])
    for item in range(len(result)):
        plt.plot(result[item]*100, label=signals[item])
    plt.legend(loc='best')
    plt.savefig(join(fig_folder, "accuracy-" + loaded_model_id + ".png"),
                bbox_inches='tight', dpi=300)


def confusion_matrix(input_model, input_model_id):
    # Here we make a prediction using the test data...
    print('\nStarting prediction')
    predict = np.argmax(input_model.predict(X_test_2), axis=-1)

    # And create a Confusion Matrix for a better visualization!
    print('\nConfusion Matrix:')
    confusion_matrix = tf.math.confusion_matrix(y_test_2, predict).numpy()
    confusion_matrix_normalized = np.around(
        confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
        decimals=2)
    print(confusion_matrix)
    print(confusion_matrix_normalized)
    cm_data_frame = pd.DataFrame(confusion_matrix_normalized, index=signals, columns=signals)
    figure = plt.figure(figsize=(8, 4), dpi=150)
    sns.heatmap(cm_data_frame, annot=True, cmap=plt.cm.get_cmap('Blues', 6))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(fig_folder, 'confusion_matrix-' + input_model_id + '_' + str(len(training_SNR)) + '.png'),
                bbox_inches='tight', dpi=300)


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
    if len(data) == number_of_used_features * 2:
        num_array = np.zeros((number_of_used_features,), dtype=np.float32)
        counter_array = np.zeros((number_of_used_features,), dtype=np.int32)
    else:
        num_array = np.zeros((len(data) - 1,), dtype=np.float32)
        counter_array = np.zeros((1,), dtype=np.int32)
    if len(data) == 2:  # Prediction
        print('Prediction')
        new_data = b''.join(data)
        num_array = struct.unpack('<f', new_data[0:4])
        counter_array = struct.unpack('<i', new_data[4:8])
    elif len(data) == number_of_used_features * 2:  # Features and counters
        print('Features')
        new_data = b''.join(data)
        for x in range(0, number_of_used_features):
            aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
            num_array[x] = aux[0]
        for x in range(number_of_used_features, number_of_used_features * 2):
            aux = struct.unpack('<i', new_data[x * 4:x * 4 + 4])
            counter_array[x - number_of_used_features] = aux[0]
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
    for x in range(0, number_of_used_features):
        aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
        np_array[x] = aux[0]
    for x in range(number_of_used_features, number_of_used_features * 2):
        aux = struct.unpack('<f', new_data[x * 4:x * 4 + 4])
        np_array[x] = aux[0]

    return np_array


def serial_communication():
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
    for point in range(0, number_of_used_features):
        binary = struct.pack('<f', np.float32(scaler.mean_[point]))
        ser.write(binary)
    for point in range(0, number_of_used_features):
        binary = struct.pack('<f', np.float32(scaler.scale_[point]))
        ser.write(binary)

    received_scaler = receive_scaler(ser, number_of_used_features * 2)
    err_scaler = received_scaler - np.concatenate((np.float32(scaler.mean_), np.float32(scaler.scale_)))
    if np.max(err_scaler) == 0:
        print('Scaler loaded successfully')
        print(received_scaler)
    else:
        print('Error loading scaler')

    print('Wait...')
    time.sleep(3)

    print('Starting modulation signals transmission!')
    for mod in signals:
        # Create error information arrays and features results from microcontroller
        err_abs_vector = []
        err_phase_vector = []
        err_unwrapped_phase_vector = []
        err_freq_vector = []
        err_cn_abs_vector = []
        err_features = []
        predictions = []
        gathered_data = []
        parsed_signal = data_mat[mat_info[mod]]
        for snr in snr_range:
            for frame in frame_range:
                print('Modulation = ' + mod)
                print('SNR = {}'.format(snr))
                print('Frame = {}'.format(frame))
                i_values = functions.InstValues(parsed_signal[snr, frame, 0:frame_size])
                ft = np.float32(features.calculate_features(parsed_signal[snr, frame, 0:frame_size]))
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


if __name__ == '__main__':
    # Filename setup
    mat_file_name = pathlib.Path(join(os.getcwd(), 'mat-data', 'all_modulations.mat'))

    # Load MAT file and parse data
    data_mat = scipy.io.loadmat(mat_file_name)
    print(str(mat_file_name) + ' file loaded...')

    training = True

    X_train, X_test, y_train, y_test, scaler = preprocess_data('Training')
    X_test_2, y_test_2 = configure_data()
    X_test_2 = scaler.transform(X_test_2)

    if training:
        # Weights and biases parser
        parser = argparse.ArgumentParser(description='RNA argument parser')
        parser.add_argument('--activation', action='store', dest='activation')
        parser.add_argument('--batch_size', action='store', dest='batch_size')
        parser.add_argument('--dropout', action='store', dest='dropout')
        parser.add_argument('--epochs', action='store', dest='epochs')
        parser.add_argument('--layer_size_hl1', action='store', dest='layer_size_hl1')
        parser.add_argument('--layer_size_hl2', action='store', dest='layer_size_hl2')
        parser.add_argument('--layer_size_hl3', action='store', dest='layer_size_hl3')
        parser.add_argument('--learning_rate', action='store', dest='learning_rate')
        parser.add_argument('--optimizer', action='store', dest='optimizer')
        args = parser.parse_args()

        wandb.init(project="amcpy-team", config=HyperParameter(None).get_dict())
        config = wandb.config
        model_id = train_rna(config)
        loaded_model, loaded_model_id = get_model_from_id(model_id)
        evaluate_rna(loaded_model)

    if not training:
        loaded_model, loaded_model_id = get_model_from_id('12e326ee')
        evaluate_ann()
        confusion_matrix(loaded_model, loaded_model_id)

        # load_dict, info_dict = quantization.quantize(loaded_model, np.concatenate((X_train, X_test)))
        # for info in info_dict:
        #     print(info + ' -> ' + info_dict[info])
        # weights = load_dict['weights']
        # biases = load_dict['biases']
        # serial_communication()
