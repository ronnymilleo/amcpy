import pickle
import sys

import scipy.io

if len(sys.argv) > 2:
    source_name = sys.argv[1]
    dest_name = sys.argv[2]

    a = pickle.load(open(source_name, "rb"))

    scipy.io.savemat(dest_name, mdict={'pickle_data': a})

    print("Data successfully converted to .mat file with variable name \"pickle_data\"")
else:
    print("Usage: pickle_to_mat_converter.py source_name.pickle mat_filename.mat")
