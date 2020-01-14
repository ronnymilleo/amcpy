from functions import gmax, mean, std, meanofsquared
import numpy as np

print("Tests")

var = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x = gmax(var)
y = mean(var)
z = std(var)
a = meanofsquared(var)

print(x)
print(y)
print(z)
print(a)
