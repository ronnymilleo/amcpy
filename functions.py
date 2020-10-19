import numpy as np
from scipy import stats


class InstValues:
    # For debug purposes only
    def __init__(self, signal_input):
        self.inst_abs = np.abs(signal_input)
        self.inst_phase = np.angle(signal_input)
        self.inst_unwrapped_phase = np.unwrap(np.angle(signal_input))
        self.inst_freq = 1 / (2 * np.pi) * np.diff(np.unwrap(np.angle(signal_input)))
        self.inst_cna = np.abs(signal_input) / np.mean(np.abs(signal_input)) - 1


class MomentValues:
    # E[x] = mean(x) when the probability p(x) is equal for every sample
    # Mpq = E[x^(p-q).x*^q]
    def __init__(self, signal_input):
        self.m20 = np.mean(signal_input ** 2)
        self.m21 = np.mean(signal_input * np.conj(signal_input))
        self.m22 = np.mean(np.conj(signal_input) ** 2)
        self.m40 = np.mean(signal_input ** 4)

        self.m41 = np.mean(signal_input * signal_input * signal_input * np.conj(signal_input))
        self.m42 = np.mean(signal_input * signal_input * np.conj(signal_input) * np.conj(signal_input))
        self.m43 = np.mean(signal_input * np.conj(signal_input) * np.conj(signal_input) * np.conj(signal_input))
        self.m60 = np.mean(np.power(signal_input, 6))
        self.m61 = np.mean(np.power(signal_input, 6 - 1) * np.power(np.conj(signal_input), 1))
        self.m62 = np.mean(np.power(signal_input, 6 - 2) * np.power(np.conj(signal_input), 2))
        self.m63 = np.mean(np.power(signal_input, 6 - 3) * np.power(np.conj(signal_input), 3))


# Std of the Instantaneous Phase
def std_dev_inst_phase(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(i.inst_phase)
    return ft_output


# Std of the Absolute Instantaneous Phase
def std_dev_abs_inst_phase(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(np.abs(i.inst_phase))
    return ft_output


# Std of the Instantaneous Frequency
def std_dev_inst_freq(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(i.inst_freq)
    return ft_output


# Std of the Absolute Instantaneous Frequency
def std_dev_abs_inst_freq(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(np.abs(i.inst_freq))
    return ft_output


# Std of the Absolute CN Instantaneous Amplitude
def std_dev_inst_cna(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(i.inst_cna)
    return ft_output


# Std of the CN Instantaneous Amplitude
def std_dev_abs_inst_cna(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(np.abs(i.inst_cna))
    return ft_output


# Mean Value of the Signal Magnitude
def mean_of_signal_magnitude(signal_input):
    ft_output = np.mean(np.abs(signal_input))
    return ft_output


# Squared Mean of the Signal Magnitude
def squared_mean_of_signal_magnitude(signal_input):
    i = InstValues(signal_input)
    ft_output = np.power(np.mean(i.inst_abs), 2)
    return ft_output


# Normalized Sqrt Value of Sum of Amplitude
def normalized_sqrt_of_sum_of_amp(signal_input):
    ft_output = np.sqrt(np.sum(np.abs(signal_input))) / len(signal_input)
    return ft_output


# Ratio of I/Q Components
def ratio_iq(signal_input):
    ft_output = np.sum(np.power(np.imag(signal_input), 2)) / np.sum(np.power(np.real(signal_input), 2))
    return ft_output


# Gmax
def gmax(signal_input):
    ft_output = np.max(np.power(np.abs(np.fft.fft(signal_input)), 2) / len(signal_input))
    return ft_output


# Kurtosis of the Absolute Amplitude
def kurtosis_of_abs_amplitude(signal_input):
    ft_output = stats.kurtosis(np.abs(signal_input))
    return ft_output


# Kurtosis of the Absolute Frequency
def kurtosis_of_abs_freq(signal_input):
    ft_output = stats.kurtosis(np.abs(1 / (2 * np.pi) * np.diff(np.unwrap(np.angle(signal_input)))))
    return ft_output


def cumulant_20(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m20)


def cumulant_21(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m21)


def cumulant_40(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m40 - 3 * m.m20 * m.m20)


def cumulant_41(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m41 - 3 * m.m20 * m.m21)


def cumulant_42(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m42 - np.power(np.abs(m.m20), 2) - 2 * np.power(m.m21, 2))


def cumulant_60(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m60 - 15 * m.m20 * m.m40 + 3 * np.power(m.m20, 3))


def cumulant_61(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m61 - 5 * m.m21 * m.m40 - 10 * m.m20 * m.m41 + 30 * np.power(m.m20, 2) * m.m21)


def cumulant_62(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m62 - 6 * m.m20 * m.m42 - 8 * m.m21 * m.m41 - m.m22 * m.m40 + \
                  6 * np.power(m.m20, 2) * m.m22 + 24 * np.power(m.m21, 2) * m.m20)


def cumulant_63(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m63 - 9 * m.m21 * m.m42 + 12 * np.power(m.m21, 3) - 3 * m.m20 * m.m43 - \
                  3 * m.m22 * m.m41 + 18 * m.m20 * m.m21 * m.m22)
