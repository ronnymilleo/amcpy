import h5py
import numpy as np
import pickle
import pathlib
import json

sure = input("\nWARNING!This code uses a high amount of RAM memory - up to 20GB. Are you sure you want to continue? Y/N ")

if not (sure == "N" or sure == "n") and (sure == "Y" or sure == "y"):
    dataset = pathlib.Path("../dataset/201801a/GOLD_XYZ_OSC.0001_1024.hdf5")
    classes = pathlib.Path("../amcPython/classes.txt")
    snr = pathlib.Path("../amcPython/snr.txt")

    #Extract the datasets from the hdf5 file
    #X=I/Q Modulation data | Y=Modulation | Z=SNR
    with h5py.File(dataset, "r") as f:
        print("Keys: %s" %f.keys())
        print("Values: %s" %f.values())
        print("Names: %s" %f.name)
        keys = list(f.keys())    
        
        dset_X = f['X']    
        data_X = dset_X[:]
        dset_Y = f['Y']
        dset_Z = f['Z']

    #Read the classes and snr txt 
    with open(classes, 'r') as f:
        d = np.loadtxt(f, dtype='str')

    with open(snr, 'r') as f:
        s = np.loadtxt(f, dtype='str')

    #Creates a dict containing an index information for locating a combination of 
    #modulation and SNR and save it to a JSON file called 'dataset_index'
    index = {}
    for idx in range(len(d)):
        location = idx * 106496
        for snr in range(len(s)):
            index[d[idx].rstrip() + ":" + s[snr].rstrip()] = str((snr * 4096) + location) + ":" + str(((snr * 4096) + 4095) + location)

    with open('dataset_index.json', 'w') as handle:
        handle.write(json.dumps(index))

    #Stores the data extracted in a pickle file with name 'MOD_SNR.pickle'
    for mod in d:
        for snr in s:
            f = mod + "_" + snr + ".pickle"
            with open(f, 'wb') as handle:
                pickle.dump(data_X[int(index[mod + ":" + snr].split(":")[0]):int(index[mod + ":" + snr].split(":")[1])], handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Execution stopped by the user.")