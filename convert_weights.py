import h5py
import pathlib
import os
from os.path import join
import numpy as np

path = "C:\\Users\\adeni\\Documents\\Mestrado\\amcpy\\rna"
weights = "weights-96ec22de.h5"

with h5py.File(join(path,weights), "r") as f:
    main_groups = f.keys()
    print("RNA layers: {}".format(main_groups))

    for group in main_groups:
        layer_group = f[group].keys()
        if len(layer_group) == 0:
            print("Check WANDB for dropout information.")
        else:
            for layer in layer_group:
                layer_keys = f[group][layer].keys()            

            for key in layer_keys:
                np.savetxt("{}_{}.csv".format(layer, key.split(":")[0]), f[group][layer][key].value, delimiter=",")
