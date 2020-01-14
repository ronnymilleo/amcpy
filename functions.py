import math

import numpy


def gmax(inputarray):
    fftsquared = abs(numpy.fft.fft(inputarray)) ** 2
    psd = fftsquared / len(inputarray)
    output = max(psd)
    return output


def mean(a):
    b = sum(a) / len(a)
    return b


def meanofsquared(a):
    aux1 = a ** 2
    aux2 = sum(aux1)
    out = aux2 / len(a)
    return out


def std(input):
    aux1 = (input - mean(input))
    aux2 = aux1 ** 2
    aux3 = sum(aux2)
    aux4 = 1 / (len(input) - 1)
    output = math.sqrt(aux3 * aux4)
    return output
