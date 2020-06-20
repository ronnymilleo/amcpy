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


def instantaneous_unwrapped_phase(signal_input):
    output = np.unwrap(np.angle(signal_input))
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


def moment(input, p, q):
    aux = input ** (p - q)
    aux2 = np.conj(input)
    aux3 = aux2 ** q
    aux4 = sum(aux * aux3)
    aux5 = 1/len(input)
    output = aux5 * aux4
    return output


def cum20(input):
    aux = input ** 2
    aux1 = sum(aux)
    aux2 = 1/len(input)
    output = aux2 * aux1
    return abs(output)


def cum21(input):
    aux = abs(input) ** 2
    aux1 = sum(aux)
    aux2 = 1/len(input)
    output = aux2 * aux1
    return output


def cum40(input):
    '''
    aux = input ** 4
    aux2 = cum20(input)
    aux3 = aux2 ** 2
    aux4 = 3 * aux3
    aux5 = aux - aux4
    aux6 = sum(aux5)
    aux7 = 1/len(input)
    output = aux7 * aux6
    '''
    output = moment(input, 4, 0) - (3 * (moment(input, 2, 0) ** 2))
    return abs(output)


def cum41(input):
    '''
    c20 = cum20(input)
    c21 = cum21(input)
    aux = 3 * c20 * c21
    aux1 = input ** 3
    aux2 = np.conj(input)
    aux3 = aux1 * aux2
    aux4 = aux3 - aux
    aux5 = sum(aux4)
    aux6 = 1/len(input)
    output = aux5 * aux6
    '''
    output = moment(input, 4, 1) - \
        (3 * moment(input, 2, 1) * moment(input, 2, 0))
    return abs(output)


def cum42(input):
    '''
    c20 = abs(cum20(input)) ** 2
    aux = cum21(input) ** 2
    c21 = 2 * aux
    aux1 = abs(input) ** 4
    aux2 = aux1 - c20 - c21
    aux3 = sum(aux2)
    aux4 = 1/len(input)
    output = aux3 * aux4
    '''
    output = moment(input, 4, 2) - (abs(moment(input, 2, 0))
                                    ** 2) - (2 * (moment(input, 2, 1) ** 2))
    return abs(output)


def cum60(input):
    ''' 
    input_6th = input ** 6
    input_4th = input ** 4
    cum_rd = cum20(input) ** 3

    term2 = 15 * cum20(input) * input_4th
    term3 = 3 * cum_rd

    aux = sum(input_6th - term2 + term3)
    aux2 = 1/len(input)
    output = aux2 * aux
    '''
    output = moment(input, 6, 0) - (15 * moment(input, 2, 0) *
                                    moment(input, 4, 0)) + (3 * (moment(input, 2, 0) ** 3))
    return abs(output)


def cum61(input):
    '''
    input_conj = np.conj(input)
    input_rd = input ** 3
    input_4th = input ** 4
    input_5th = input ** 5
    cum_nd = cum20(input) ** 2

    term1 = input_5th * input_conj
    term2 = 5 * cum21(input) * input_4th
    term3 = 10 * cum20(input) * input_rd * input_conj
    term4 = 30 * cum_nd * cum21(input)

    aux = sum(term1 - term2 - term3 + term4)
    aux2 = 1/len(input)
    output = aux2 * aux
    '''
    output = moment(input, 6, 1) - (5 * moment(input, 2, 1) * moment(input, 4, 0)) - (10 * moment(
                    input, 2, 0) * moment(input, 4, 1)) + (30 * (moment(input, 2, 0) ** 2) * moment(input, 2, 0))
    return abs(output)


def cum62(input):
    '''
    input_nd = input ** 2
    input_rd = input ** 3
    input_4th = input ** 4
    input_conj = np.conj(input)
    input_conj_nd = np.conj(input) ** 2
    cum20_nd = cum20(input) ** 2
    cum21_nd = cum21(input) ** 2

    term1 = input_4th * input_conj_nd
    term2 = 6 * cum20(input) * input_nd * input_conj_nd
    term3 = 8 * cum21(input) * input_rd * input_conj
    term4 = input_conj_nd * input_4th
    term5 = 6 * cum20_nd * input_conj_nd
    term6 = 24 * cum21_nd * cum20(input)

    aux = sum(term1 - term2 - term3 - term4 + term5 + term6)
    aux2 = 1/len(input)
    output = aux2 * aux
    '''
    output = moment(input, 6, 2) - (6 * moment(input, 2, 0) * moment(input, 4, 2)) - \
                    (8 * moment(input, 2, 1) * moment(input, 4, 1)) - (moment(input, 2, 2) * \
                    moment(input, 4, 0)) + (6 * (moment(input, 2, 0) ** 2) * moment(input, 2, 2)) + \
                    (24 * (moment(input, 2, 1) ** 2) * moment(input, 2, 0))
    return abs(output)


def cum63(input):
    '''
    input_nd = input ** 2
    input_rd = input ** 3
    input_conj = np.conj(input)
    input_conj_nd = np.conj(input) ** 2
    input_conj_rd = np.conj(input) ** 3
    cum21_rd = cum21(input) ** 3

    term1 = input_rd * input_conj_rd
    term2 = 9 * cum21(input) * input_nd * input_conj_nd
    term3 = 12 * cum21_rd
    term4 = 3 * cum20(input) * input * input_conj_rd
    term5 = 3 * input_conj_nd * input_rd * input_conj
    term6 = 18 * cum20(input) * cum21(input) * input_conj_nd

    aux = sum(term1 - term2 + term3 - term4 - term5 + term6)
    aux2 = 1/len(input)
    output = aux2 * aux
    '''
    output = moment(input, 6, 3) - (9 * moment(input, 2, 1) * moment(input, 4, 2)) + \
                    (12 * (moment(input, 2, 1) ** 3)) - (3 * moment(input, 2, 0) * moment(input, 4, 3)) - \
                    (3 * moment(input, 2, 2) * moment(input, 4, 1)) + (18 * moment(input, 2, 0) * \
                    moment(input, 2, 1) * moment(input, 2, 2))
    return abs(output)


def meanAbsolute(input):
    aux = instantaneous_absolute(input)
    aux2 = sum(aux)
    aux3 = 1/len(input)
    output = aux3 * aux2
    return output


def sqrtAmplitude(input):
    aux = instantaneous_absolute(input)
    aux2 = sum(aux)
    aux3 = math.sqrt(aux2)
    output = aux3 / len(input)
    return output


def ratioIQ(input):
    aux = (input.real) ** 2
    aux1 = sum(aux)
    aux2 = (input.imag) ** 2
    aux3 = sum(aux2)
    output = aux3/aux1
    return output