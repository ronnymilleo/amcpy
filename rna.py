import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
import pathlib
from os.path import isfile, join
import os
import pickle

number_of_frames = 500
number_of_features = 22
number_of_snr = 9

print("\nBE READY, THE MAGIC IS ABOUT TO BEGIN!\n")

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
        print("")
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

#Is it necessary to normalize the data?
dataTrain, dataTest, targetTrain, targetTest = train_test_split(dataRna, target, test_size=0.3)
print(dataTrain.shape, dataTest.shape, targetTrain.shape, targetTest.shape)

model=Sequential()
model.add(Dropout(0.1, input_shape=(dataTrain.shape[1],)))
model.add(Dense(22, activation="relu", kernel_initializer="he_normal"))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit(dataTrain, targetTrain, epochs=150, verbose=1)

loss, acc = model.evaluate(dataTest, targetTest, verbose=1)
print('Test Accuracy: %.3f' % acc)