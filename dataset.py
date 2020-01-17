import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset = "/home/Adenilson.Castro/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
datasetRonny = "C:\\Users\\ronny\\Dataset\\GOLD_XYZ_OSC.0001_1024.hdf5"

#Extract the datasets from the hdf5 file
#X=I/Q Modulation data - Y=Modulation - Z=SNR
with h5py.File(datasetRonny, "r") as f:
    print("Keys: %s" %f.keys())
    print("Values: %s" %f.values())
    print("Names: %s" %f.name)
    keys = list(f.keys())    
    
    dset_X = f['X']
    data_X = dset_X[2490368:2490468] #Extract 100 frames from 16QAM with 0dB of SNR

    dset_Y = f['Y']
    dset_Z = f['Z']

temp = []
for sample in range(len(data_X)):
    for idx in range(len(data_X[sample])):        
        temp.append(data_X[sample][idx])

#Save the samples ina pickle file
with open('16qam.pickle', 'wb') as handle:
    pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Load the pickle file
with open('16qam.pickle', 'rb') as handle:
    data = pickle.load(handle)

#TODO: Improve the data extraction