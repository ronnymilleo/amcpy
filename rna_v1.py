import argparse
import json
import os
import pathlib
import pickle
import time
import uuid
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import wandb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from wandb.keras import WandbCallback

# Loads JSON file with execution setup
with open("./info.json") as handle:
    info_json = json.load(handle)

# Config variables based on the JSON file
number_of_frames = info_json['numberOfFrames']
number_of_features = len(info_json['features']['using'])
number_of_snr = len(info_json['snr']['using'])
snr_list = info_json['snr']['using']
modulations = info_json['modulations']['names']

# Explicitly selects the dataset for TRAINING the RNA
if info_json['dataSetForTraining'] == "awgn":
    features_files = [f + "_awgn_features.pickle" for f in modulations]
elif info_json['dataSetForTraining'] == "rayleigh":
    features_files = [f + "_features.pickle" for f in modulations]
elif info_json['dataSetForTraining'] == "matlab":
    features_files = [f + "_features.pickle" for f in modulations]

# Explicitly selects the dataset for TRAINING the RNA
if info_json['dataSetForTraining'] == "awgn":
    test_data_files = [f + "_awgn_features.pickle" for f in modulations]
elif info_json['dataSetForTraining'] == "rayleigh":
    test_data_files = [f + "_features.pickle" for f in modulations]
elif info_json['dataSetForTraining'] == "matlab":
    test_data_files = [f + "_features.pickle" for f in modulations]


def process_data():  # Prepare the data for the magic
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))
    data_rna = np.zeros((number_of_frames * number_of_snr * len(features_files), number_of_features))
    target = []
    samples = number_of_frames * number_of_snr

    # Here each modulation file is loaded and all
    # frames to all SNR values are vertically stacked
    for i, mod in enumerate(features_files):
        print("Processing {} data".format(mod.split("_")[0]))
        with open(join(data_folder, mod), 'rb') as ft_handle:
            data = pickle.load(ft_handle)

        location = i * number_of_frames * number_of_snr

        for snr in snr_list:
            for frame in range(info_json['numberOfFrames']):
                data_rna[location, :] = data[snr - (21 - number_of_snr)][frame][:]
                location += 1

        # An array containing the labels for
        # each modulation is then created...
        start = i * samples
        end = start + samples
        for _ in range(start, end):
            target.append(mod.split("_")[0])

    # ...and encoded to labels ranging from 0 to 4 - 4 modulations + noise
    for mod in modulations:
        for item in range(len(target)):
            if target[item] == mod:
                target[item] = info_json['modulations']['labels'][mod]
    target = np.asarray(target)

    # Finally, the data is split into train and test
    # samples and normalized for a better learning
    data_train, data_test, target_train, target_test = train_test_split(data_rna, target, test_size=0.3)
    print("\nData shape:")
    print(data_train.shape, data_test.shape, target_train.shape, target_test.shape)
    data_train_norm = normalize(data_train, norm='l2')
    data_test_norm = normalize(data_test, norm='l2')

    return data_train_norm, data_test_norm, target_train, target_test


def train_rna(config):
    data_train, data_test, target_train, target_test = process_data()
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    id = str(uuid.uuid1()).split('-')[0]  # Generates a unique id to each RNA created

    # Here is where the magic really happens! Check this out:
    model = Sequential()  # The model used is the sequential
    # It has a fully connected input layer
    model.add(Dense(data_train.shape[1], activation="relu", kernel_initializer=config.initializer,
                    input_shape=(data_train.shape[1],)))
    # With three others hidden layers
    model.add(Dense(config.layer_size_hl1, activation=config.activation, kernel_initializer=config.initializer))
    # And a dropout layer between them
    model.add(Dropout(config.dropout))
    model.add(Dense(config.layer_size_hl2, activation=config.activation, kernel_initializer=config.initializer))
    model.add(Dropout(config.dropout))
    model.add(Dense(config.layer_size_hl3, activation=config.activation, kernel_initializer=config.initializer))
    model.add(Dense(len(modulations), activation='softmax'))

    # Once created, the model is then compiled, trained
    # and saved for further evaluation
    model.compile(optimizer=config.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(data_train, target_train, validation_split=0.3, epochs=config.epochs, verbose=1,
                        callbacks=[WandbCallback(validation_data=(data_test, target_test))])
    
    model.save(str(join(rna_folder, 'rna-' + id + '.h5')))
    model.save_weights(str(join(rna_folder, 'weights-' + id + '.h5')))
    print(join("\nRNA saved with id ", id, "\n").replace("\\", ""))

    # A figure with a model representation is automatically saved!
    plot_model(model, to_file=join(fig_folder, 'model-' + id + '.png'), show_shapes=True)

    # Here is where we make the first evaluation of the RNA
    loss, acc = model.evaluate(data_test, target_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)

    # Here, WANDB takes place and logs all metrics to the cloud
    metrics = {'accuracy': acc,
               'loss': loss,
               'dropout': config.dropout,
               'epochs': config.epochs,
               'initializer': config.initializer,
               'layer_syze_hl1': config.layer_size_hl1,
               'layer_syze_hl2': config.layer_size_hl2,
               'layer_syze_hl3': config.layer_size_hl3,
               'optimizer': config.optimizer,
               'activation': config.activation,
               'id': id}
    wandb.log(metrics)

    # Here we make a prediction using the test data...
    print('\nStarting prediction')
    predict = model.predict_classes(data_test, verbose=1)

    # And create a Confusion Matrix for a better visualization!
    print('\nConfusion Matrix:')
    confusion_matrix = tf.math.confusion_matrix(target_test, predict).numpy()
    confusion_matrix_normalized = np.around(
        confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
        decimals=2)
    print(confusion_matrix_normalized)
    cm_data_frame = pd.DataFrame(confusion_matrix_normalized, index=modulations, columns=modulations)
    figure = plt.figure(figsize=(8, 4), dpi=150)
    sns.heatmap(cm_data_frame, annot=True, cmap=plt.cm.get_cmap('Blues', 6))
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(fig_folder, 'confusion_matrix-' + id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_accuracy-' + id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_loss-' + id + '.png'), bbox_inches='tight', dpi=300)

    plt.close(figure)
    evaluate_rna(id=id)


def evaluate_rna(id="foo", test_size=500):  # Make a prediction using some samples to evaluate the RNA behavior
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))

    print("\nStarting RNA evaluation by SNR.")

    if id == "foo":  # If you do not specify a RNA id, it'll use the newest available in rna_folder
        aux = [f for f in os.listdir(rna_folder) if "rna" in f]
        rna_files = [join(str(rna_folder), item) for item in aux]
        latest_rna_model = max(rna_files, key=os.path.getctime)
        print("\nRNA ID not provided. Using RNA model with id {}, created at {} instead.".format(
            latest_rna_model.split("-")[1].split(".")[0], time.ctime(os.path.getmtime(latest_rna_model))))

        model = load_model(latest_rna_model)  # Loads the RNA model

        # For each modulation, radomnly loads the test_size samples
        # and predict the result to all SNR values
        result = np.zeros((len(modulations), number_of_snr))
        for i, mod in enumerate(test_data_files):
            print("Evaluating {}".format(mod.split("_")[0]))
            with open(join(data_folder, mod), 'rb') as evaluating_data:
                data = pickle.load(evaluating_data)
            for j, snr in enumerate(snr_list):
                random_samples = np.random.choice(data[snr][:].shape[0], test_size)
                data_test = [data[snr][i] for i in random_samples]
                data_test = normalize(data_test, norm='l2')
                right_label = [info_json['modulations']['labels'][mod.split("_")[0]] for _ in range(len(data_test))]
                predict = model.predict_classes(data_test)
                accuracy = accuracy_score(right_label, predict)
                result[i][snr] = accuracy

        # Then, it creates an accuracy graphic, containing the
        # prediction result to all snr values and all modulations
        figure = plt.figure(figsize=(8, 4), dpi=150)
        plt.title("Accuracy")
        plt.ylabel("Right prediction")
        plt.xlabel("SNR [dB]")
        plt.xticks(np.arange(number_of_snr), [info_json['snr']['values'][i] for i in info_json['snr']['using']])
        for item in range(len(result)):
            plt.plot(result[item], label=modulations[item])
        plt.legend(loc='best')
        plt.savefig(join(fig_folder, "accuracy-" + latest_rna_model.split("-")[1].split(".")[0] + ".png"),
                    bbox_inches='tight', dpi=300)

        figure.clf()
        plt.close(figure)
    else:  # If you specify a RNA id, it will use it and make the exact same steps as the previous one
        rna = join(str(rna_folder), "rna-" + id + ".h5")
        model = load_model(rna)
        print("\nUsing RNA with id {}.".format(id))

        result = np.zeros((len(modulations), number_of_snr))
        for i, mod in enumerate(test_data_files):
            print("Evaluating {}".format(mod.split("_")[0]))
            with open(join(data_folder, mod), 'rb') as evaluating_data:
                data = pickle.load(evaluating_data)
            for j, snr in enumerate(snr_list):
                random_samples = np.random.choice(data[snr - (21 - number_of_snr)][:].shape[0], test_size)
                data_test = [data[snr - (21 - number_of_snr)][i] for i in random_samples]
                data_test = normalize(data_test, norm='l2')
                right_label = [info_json['modulations']['labels'][mod.split("_")[0]] for _ in range(len(data_test))]
                predict = model.predict_classes(data_test)
                accuracy = accuracy_score(right_label, predict)
                result[i][j] = accuracy

        figure = plt.figure(figsize=(8, 4), dpi=150)
        plt.title("Accuracy")
        plt.ylabel("Right prediction")
        plt.xlabel("SNR [dB]")
        plt.xticks(np.arange(number_of_snr), [info_json['snr']['values'][i] for i in info_json['snr']['using']])
        for item in range(len(result)):
            plt.plot(result[item], label=modulations[item])
        plt.legend(loc='best')
        plt.savefig(join(fig_folder, "accuracy-" + id + ".png"), bbox_inches='tight', dpi=300)

        figure.clf()
        plt.close(figure)

        if not os.path.isfile(join(rna_folder, "weights-" + id + ".h5")):
            print("Weights file not found. Saving it into RNA folder")
            model.save_weights(str(join(rna_folder, 'weights-' + id + '.h5')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNA argument parser')
    parser.add_argument('--dropout', action='store', dest='dropout')
    parser.add_argument('--epochs', action='store', dest='epochs')
    parser.add_argument('--optimizer', action='store', dest='optimizer')
    parser.add_argument('--initializer', action='store', dest='initializer')
    parser.add_argument('--layer_size_hl1', action='store', dest='layer_size_hl1')
    parser.add_argument('--layer_size_hl2', action='store', dest='layer_size_hl2')
    parser.add_argument('--layer_size_hl3', action='store', dest='layer_size_hl3')
    parser.add_argument('--activation', action='store', dest='activation')
    arguments = parser.parse_args()

    # WANDB hyperparameters setup
    hyperparameterDefaults = dict(
        dropout=round(float(arguments.dropout), 2),
        epochs=int(arguments.epochs),
        optimizer=arguments.optimizer,
        activation=arguments.activation,
        initializer=arguments.initializer,
        layer_size_hl1=int(arguments.layer_size_hl1),
        layer_size_hl2=int(arguments.layer_size_hl2),
        layer_size_hl3=int(arguments.layer_size_hl3)
    )
    wandb.init(project="amcpy-team", config=hyperparameterDefaults)
    config = wandb.config

    #evaluate_rna(id="96ec22de")
    train_rna(config)
