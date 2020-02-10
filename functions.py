import math

import numpy as np


def gmax(signal_input):
    abs_fft = abs(np.fft.fft(signal_input))
    abs_fft_squared = abs_fft ** 2
    psd = abs_fft_squared / len(signal_input)
    output = max(psd)
    return output


def mean(signal_input):
    output = sum(signal_input) / len(signal_input)
    return output


def mean_of_squared(signal_input):
    aux1 = signal_input ** 2
    aux2 = sum(aux1)
    output = aux2 / len(signal_input)
    return output


def std_deviation(signal_input):
    aux1 = (signal_input - mean(signal_input))
    aux2 = aux1 ** 2
    aux3 = sum(aux2)
    aux4 = 1 / (len(signal_input) - 1)
    output = math.sqrt(aux3 * aux4)
    return output


def kurtosis(signal_input):
    m = mean(signal_input)
    aux4 = (signal_input - m) ** 4
    aux2 = (signal_input - m) ** 2
    num = (1 / len(signal_input)) * sum(aux4)
    den = ((1 / len(signal_input)) * sum(aux2)) ** 2
    output = num / den
    return output


def instantaneous_phase(signal_input):
    output = np.angle(signal_input)
    return output


def instantaneous_frequency(signal_input):
    output = 1 / (2 * np.pi) * np.diff(np.unwrap(np.angle(signal_input)))
    return output


def instantaneous_absolute(signal_input):
    output = abs(signal_input)
    return output


def instantaneous_cn_absolute(signal_input):
    output = abs(signal_input) / mean(abs(signal_input)) - 1
    return output
