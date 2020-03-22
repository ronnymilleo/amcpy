import os
import pathlib
import pickle
from os.path import join

import h5py

dataset = "/home/Adenilson.Castro/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
datasetRonny = "F:\\2018.01\\GOLD_XYZ_OSC.0001_1024.hdf5"

classes = [('32PSK', 1),
           ('16APSK', 2),
           ('32QAM', 3),
           ('FM', 4),
           ('GMSK', 5),
           ('32APSK', 6),
           ('OQPSK', 7),
           ('8ASK', 8),
           ('BPSK', 9),
           ('8PSK', 10),
           ('AM-SSB-SC', 11),
           ('4ASK', 12),
           ('16PSK', 13),
           ('64APSK', 14),
           ('128QAM', 15),
           ('128APSK', 16),
           ('AM-DSB-SC', 17),
           ('AM-SSB-WC', 18),
           ('64QAM', 19),
           ('QPSK', 20),
           ('256QAM', 21),
           ('AM-DSB-WC', 22),
           ('OOK', 23),
           ('16QAM', 24)]

# Choose modulation
modulation = int(input('Enter modulation number (1 ... 24): '))
print('Modulation = ' + classes[modulation - 1][0])
while (modulation < 1) or (modulation > 24):
    modulation = int(input('Try again: enter modulation number (1 ... 24): '))

# Start and end of each modulation
modulation_end = classes[modulation - 1][1] * 106496
modulation_start = modulation_end - 106496
print('Modulation start on dataset = {0}'.format(modulation_start))
print('Modulation end on dataset = {0}'.format(modulation_end))
temp = []
# Extract the dataset from the hdf5 file
# X=I/Q Modulation data - Y=Modulation - Z=SNR
with h5py.File(datasetRonny, "r") as f:
    print("Keys: %s" % f.keys())
    print("Values: %s" % f.values())
    print("Names: %s" % f.name)
    keys = list(f.keys())
    dataset_X = f['X']
    data_X = dataset_X[modulation_start:modulation_end]  # Extract 4096 frames from one SNR

temp.append(data_X)  # Append frames of every SNR

# Create file name
name = classes[modulation - 1][0] + '_RAW.pickle'

# Save the samples in a pickle file
with open(pathlib.Path(join(os.getcwd(),'data', name)), 'wb') as handle:
    pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)
