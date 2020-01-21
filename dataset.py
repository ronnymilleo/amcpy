import pickle

import h5py

dataset = "/home/Adenilson.Castro/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
datasetRonny = "C:\\Users\\ronny\\Dataset\\GOLD_XYZ_OSC.0001_1024.hdf5"

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
while (modulation < 1) or (modulation > 24):
    modulation = int(input('Try again: enter modulation number (1 ... 24): '))

# Start and end of each modulation
modulation_end = classes[modulation - 1][1] * 106496
modulation_start = modulation_end - 106496

# SNR dB select
modulationSNR = int(input('Enter modulation SNR, only pairs are valid (-20 ... 30): '))
while (modulationSNR % 2) or (modulation < -20) or (modulation > 30) != 0:
    modulationSNR = int(input('Try again: enter modulation SNR, only pairs are valid (-20 ... 30): '))

# Modulation SNR start and end config (default = 4096 frames)
SNR_end = modulation_start + (modulationSNR // 2 + 11) * 4096
SNR_start = SNR_end - 4096

# Extract the dataset from the hdf5 file
# X=I/Q Modulation data - Y=Modulation - Z=SNR
with h5py.File(datasetRonny, "r") as f:
    print("Keys: %s" % f.keys())
    print("Values: %s" % f.values())
    print("Names: %s" % f.name)
    keys = list(f.keys())

    dset_X = f['X']
    data_X = dset_X[SNR_start:SNR_end]  # Extract 4096 frames from 16QAM with 10dB of SNR

    dset_Y = f['Y']
    dset_Z = f['Z']

temp = []
for sample in range(len(data_X)):
    for idx in range(len(data_X[sample])):
        temp.append(data_X[sample][idx])

# Create file name
name = classes[modulation - 1][0] + '_SNR' + str(modulationSNR) + 'dB' + '.pickle'

# Save the samples ina pickle file
with open(name, 'wb') as handle:
    pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the pickle file
with open(name, 'rb') as handle:
    data = pickle.load(handle)
