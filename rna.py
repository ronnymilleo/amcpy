import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from os.path import isfile, join
import os
import pickle
import time

import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split

number_of_frames = 1024
number_of_features = 22
number_of_snr = 9
modulations = ['BPSK', 'QPSK', 'PSK8', 'QAM16']

def processData():
    dataFolder = pathlib.Path(join(os.getcwd(), "gr-data"))
    featuresFiles = [f for f in os.listdir(dataFolder) if "features" in f]

    dataRna = np.zeros((number_of_frames*number_of_snr*len(featuresFiles), number_of_features))
    target = []

    for i,mod in enumerate(featuresFiles):
        data = []

        with open(join(dataFolder, mod), 'rb') as handle:
            data = pickle.load(handle)

        location = i * number_of_frames * number_of_snr
        
        for snr in range(len(data)):
            for frame in range(len(data[snr])):
                dataRna[location][:] = data[snr][frame][:]
                location += 1

    samples = number_of_frames*number_of_snr
    for i, mod in enumerate(featuresFiles):
        start = i*samples
        end = start + samples
        for sample in range(start, end):
            target.append(mod.split("_")[0])

    target = LabelEncoder().fit_transform(target)

    dataTrain, dataTest, targetTrain, targetTest = train_test_split(dataRna, target, test_size=0.3)
    print(dataTrain.shape, dataTest.shape, targetTrain.shape, targetTest.shape)
    dataTrainNorm = normalize(dataTrain, norm='l2')
    dataTestNorm = normalize(dataTest, norm='l2')

    return dataTrainNorm, dataTestNorm, targetTrain, targetTest

def trainRna(dataTrain, dataTest, targetTrain, targetTest):
    rnaFolder = pathlib.Path(join(os.getcwd(), 'rna', 'rna.h5'))
    if not os.path.isfile(rnaFolder):
        model=Sequential()
        model.add(Dense(22, activation="relu", kernel_initializer="he_normal", input_shape=(dataTrain.shape[1],)))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.3))
        model.add(Dense(4, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        model.fit(dataTrain, targetTrain, epochs=100, verbose=1)
        
        model.save(str(rnaFolder))
        print("\nRNA saved.\n")

        loss, acc = model.evaluate(dataTest, targetTest, verbose=1)
        print('Test Accuracy: %.3f' % acc)
    else:
        print("\nWarning! Using an existing RNA stored on \\rna folder.\n")
        model = load_model(rnaFolder)
        loss, acc = model.evaluate(dataTest, targetTest, verbose=1)
        print('Test Accuracy: %.3f' % acc)

        print('Starting prediction')
        predict = model.predict_classes(dataTest, verbose=1)

        print('\nConfusion Matrix:\n')
        confusionMatrix = tf.math.confusion_matrix(targetTest, predict).numpy()
        confusionMatrixNormalized = np.around(confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis], decimals=2)
        print(confusionMatrixNormalized)
        
        cmDataFrame = pd.DataFrame(confusionMatrixNormalized, index=modulations, columns=modulations)

        figure = plt.figure(figsize=(8, 4),dpi=150)
        sns.heatmap(cmDataFrame, annot=True,cmap=plt.cm.get_cmap('Blues', 6))
        plt.tight_layout()
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

if __name__ == '__main__':
    dataTrain, dataTest, targetTest, targetTrain = processData()
    trainRna(dataTrain, dataTest, targetTest, targetTrain)