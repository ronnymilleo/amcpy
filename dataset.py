import h5py
import time
import numpy as np

dataset = "/home/Adenilson.Castro/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"

start = time.time()

with h5py.File(dataset, "r") as f:
    print("Keys: %s" %f.keys())
    print("Values: %s" %f.values())
    print("Names: %s" %f.name)
    keys = list(f.keys())    
    dset_X = f['X']
    data_X = dset_X[0:4095]
    #data = list(f[keys])

end = time.time()
print(end - start)