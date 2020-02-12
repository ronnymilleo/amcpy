import numpy as np
import matplotlib.pyplot as plt

file_name = "/home/gics-administrador/Documentos/file"

# with open(file_name, "rb", buffering=0) as handle:

data = np.fromfile(file_name, dtype=complex)

plt.plot(data[1:1000])