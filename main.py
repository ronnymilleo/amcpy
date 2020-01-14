import numpy as np

import features
import functions

print("Tests")

var = np.array([0, -1, 2, -3, 4, -5, 6, -7, 8, -9])

print(functions.mean(var))
print(functions.std_deviation(var))

print(functions.gmax(var))
print(functions.mean_of_squared(var))
print(functions.kurtosis(var))

print(functions.instantaneous_absolute(var))
print(functions.instantaneous_cn_absolute(var))
print(functions.instantaneous_phase(var))
print(functions.instantaneous_frequency(var))

print(features.calculate_features())
