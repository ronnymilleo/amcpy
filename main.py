import pickle
import matplotlib.pyplot as plt
import numpy as np
import features as ft

#Load the pickle file
with open('16qam.pickle', 'rb') as handle:
    data = pickle.load(handle)

var = np.zeros(len(data), dtype=np.complex)

for i in range(len(data[:])):
    var[i] = (complex(data[i][0], data[i][1]))

plt.plot(var.real)
plt.plot(var.imag)
plt.show()

features = ft.calculate_features(var)

print('Desvio padrão do valor absoluto da componente não-linear da fase instantânea: ' + str(features[0]))
print('Desvio padrão do valor direto da componente não-linear da fase instantânea: ' + str(features[1]))
print('Desvio padrão do valor absoluto da componente não-linear da frequência instantânea: ' + str(features[2]))
print('Desvio padrão do valor direto da componente não-linear da frequência instantânea: ' + str(features[3]))
print('Curtose: ' + str(features[4]))
print('Valor máximo da DEP da amplitude instantânea normalizada e centralizada: ' + str(features[5]))
print('Média da amplitude instantânea normalizada centralizada ao quadrado: ' + str(features[6]))
print('Desvio padrão do valor absoluto da amplitude instantânea normalizada e centralizada: ' + str(features[7]))
print('Desvio padrão da amplitude instantânea normalizada e centralizada: ' + str(features[8]))
