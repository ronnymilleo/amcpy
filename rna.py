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
from sklearn.preprocessing import LabelEncoder, normalize
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from wandb.keras import WandbCallback

with open("./info.json") as handle:
    infoJson = json.load(handle)

number_of_frames = infoJson['numberOfFrames']
number_of_features = len(infoJson['features']['using'])
number_of_snr = len(infoJson['snr'])
modulations = infoJson['modulations']['names']

hyperparameterDefaults = dict(
    dropout=0.3,
    learning_rate=0.001,
    epochs=25,
    optmizer="Adam",
    activation="relu",
    layerSizeHl1=48,
    layerSizeHl2=128,
    layerSizeH1=32
)
wandb.init(project="amcpy", config=hyperparameterDefaults)
config = wandb.config


def process_data():
    data_folder = pathlib.Path(join(os.getcwd(), "gr-data", "pickle"))
    features_files = [f for f in os.listdir(data_folder) if "features" in f]

    data_rna = np.zeros((number_of_frames * number_of_snr * len(features_files), number_of_features))
    target = []

    for i, mod in enumerate(features_files):

        with open(join(data_folder, mod), 'rb') as ft_handle:
            data = pickle.load(ft_handle)

        location = i * number_of_frames * number_of_snr

        for snr in range(len(data)):
            for frame in range(len(data[snr])):
                data_rna[location][:] = data[snr][frame][:]
                location += 1

    samples = number_of_frames * number_of_snr
    for i, mod in enumerate(features_files):
        start = i * samples
        end = start + samples
        for _ in range(start, end):
            target.append(mod.split("_")[0])

    target = LabelEncoder().fit_transform(target)

    data_train, data_test, target_train, target_test = train_test_split(data_rna, target, test_size=0.3)
    print("\nData shape:")
    print(data_train.shape, data_test.shape, target_train.shape, target_test.shape)
    data_train_norm = normalize(data_train, norm='l2')
    data_test_norm = normalize(data_test, norm='l2')

    return data_train_norm, data_test_norm, target_train, target_test


def train_rna(arguments):
    data_train, data_test, target_train, target_test = process_data()
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    id = str(uuid.uuid1()).split('-')[0]

    model = Sequential()
    model.add(Dense(22, activation="relu", kernel_initializer="he_normal", input_shape=(data_train.shape[1],)))
    model.add(Dropout(float(arguments.dropout)))
    model.add(Dense(int(arguments.layerSizeHl1), activation=arguments.activation, kernel_initializer='he_normal'))
    model.add(Dropout(float(arguments.dropout)))
    model.add(Dense(int(arguments.layerSizeHl2), activation=arguments.activation, kernel_initializer='he_normal'))
    model.add(Dropout(float(arguments.dropout)))
    model.add(Dense(int(arguments.layerSizeHl3), activation=arguments.activation, kernel_initializer='he_normal'))
    model.add(Dropout(float(arguments.dropout)))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=arguments.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(data_train, target_train, validation_split=0.25, epochs=int(arguments.epochs), verbose=1,
                        callbacks=[WandbCallback(validation_data=(data_test, target_test))])
    model.save(str(join(rna_folder, 'rna-' + id + '.h5')))
    print("\nRNA saved.\n")

    plot_model(model, to_file=join(fig_folder, 'model-' + id + '.png'), show_shapes=True)

    loss, acc = model.evaluate(data_test, target_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)

    metrics = {'accuracy': acc,
               'loss': loss,
               'dropout': arguments.dropout,
               'epochs': arguments.epochs,
               'layer_syze_hl1': arguments.layerSizeHl1,
               'layer_syze_hl2': arguments.layerSizeHl2,
               'layer_syze_hl3': arguments.layerSizeHl3,
               'optimizer': arguments.optimizer,
               'activation': arguments.activation}
    wandb.log(metrics)

    print('Starting prediction')
    predict = model.predict_classes(data_test, verbose=1)

    print('\nConfusion Matrix:\n')
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


def evaluate_rna(id="foo", test_size=500):
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    data_folder = pathlib.Path(join(os.getcwd(), "gr-data", "pickle"))
    data_files = [f for f in os.listdir(data_folder) if "features" in f]

    if id == "foo":
        aux = [f for f in os.listdir(rna_folder) if "rna" in f]
        rna_files = [join(str(rna_folder), item) for item in aux]
        latest_rna_model = max(rna_files, key=os.path.getctime)
        print("RNA ID not provided. Using RNA model with id {}, created at {} instead.\n".format(
            latest_rna_model.split("-")[1].split(".")[0], time.ctime(os.path.getmtime(latest_rna_model))))

        model = load_model(latest_rna_model)

        result = np.zeros((len(infoJson['modulations']['names']), len(infoJson['snr'])))
        for i, mod in enumerate(data_files):
            with open(join(data_folder, mod), 'rb') as handle:
                data = pickle.load(handle)
            for snr in range(len(data)):
                dataTest = data[snr][:test_size]
                dataTest = normalize(dataTest, norm='l2')
                rightLabel = [infoJson['modulations']['index'][i] for _ in range(len(dataTest))]
                predict = model.predict_classes(dataTest)
                accuracy = accuracy_score(rightLabel, predict)
                result[i][snr] = accuracy

        figure = plt.figure(figsize=(8, 4), dpi=150)
        plt.title("Accuracy")
        plt.ylabel("Right prediction")
        plt.xlabel("SNR [dB]")
        plt.xticks(np.arange(len(infoJson['snr'])), infoJson['snr'])
        for item in range(len(result)):
            plt.plot(result[item], label=infoJson['modulations']['names'][item])
        plt.legend(loc='best')
        plt.savefig(join(fig_folder, "accuracy-" + latest_rna_model.split("-")[1].split(".")[0] + ".png"),
                    bbox_inches='tight', dpi=300)

        figure.clf()
        plt.close(figure)
    else:
        rna = join(str(rna_folder), "rna-" + id + ".h5")
        model = load_model(rna)
        print("Using RNA with id {}.\n".format(id))

        result = np.zeros((len(infoJson['modulations']['names']), len(infoJson['snr'])))
        for i, mod in enumerate(data_files):
            with open(join(data_folder, mod), 'rb') as handle:
                data = pickle.load(handle)
            for snr in range(len(data)):
                dataTest = data[snr][:test_size]
                dataTest = normalize(dataTest, norm='l2')
                rightLabel = [infoJson['modulations']['index'][i] for _ in range(len(dataTest))]
                predict = model.predict_classes(dataTest)
                accuracy = accuracy_score(rightLabel, predict)
                result[i][snr] = accuracy

        figure = plt.figure(figsize=(8, 4), dpi=150)
        plt.title("Accuracy")
        plt.ylabel("Right prediction")
        plt.xlabel("SNR [dB]")
        plt.xticks(np.arange(len(infoJson['snr'])), infoJson['snr'])
        for item in range(len(result)):
            plt.plot(result[item], label=infoJson['modulations']['names'][item])
        plt.legend(loc='best')
        plt.savefig(join(fig_folder, "accuracy-" + id + ".png"), bbox_inches='tight', dpi=300)

        figure.clf()
        plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNA argument parser')
    parser.add_argument('--dropout', action='store', dest='dropout')
    parser.add_argument('--epochs', action='store', dest='epochs')
    parser.add_argument('--optimizer', action='store', dest='optimizer')
    parser.add_argument('--layer_size_hl1', action='store', dest='layerSizeHl1')
    parser.add_argument('--layer_size_hl2', action='store', dest='layerSizeHl2')
    parser.add_argument('--layer_size_hl3', action='store', dest='layerSizeHl3')
    parser.add_argument('--activation', action='store', dest='activation')
    arguments = parser.parse_args()
    train_rna(arguments)
