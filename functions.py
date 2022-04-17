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


def test_inst_values():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    i = InstValues(signal)
    expected_inst_abs = np.array([0, 1.4142135623730951, 2.8284271247461903, 4.242640687119285,
                                  5.656854249492381, 7.0710678118654755, 8.485281374238570, 9.899494936611665,
                                  11.313708498984761, 12.727922061357855], dtype=np.float64)
    expected_inst_phase = np.array([0, 2.356194490192345, -0.7853981633974483, 2.356194490192345,
                                    -0.7853981633974483, 2.356194490192345, -0.7853981633974483, 2.356194490192345,
                                    -0.7853981633974483, 2.356194490192345], dtype=np.float64)
    expected_inst_freq = np.array([0.375, -0.5, 0.5, -0.5,
                                   0.5, -0.5, 0.5, -0.5,
                                   0.5], dtype=np.float64)
    expected_inst_cna = np.array([-1, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337,
                                  -0.11111111111111105, 0.11111111111111116, 0.33333333333333326, 0.5555555555555556,
                                  0.7777777777777779, 1], dtype=np.float64)
    for idx in range(0, len(signal)):
        assert i.inst_abs[idx] == expected_inst_abs[idx]
        assert i.inst_phase[idx] == expected_inst_phase[idx]
        if idx < len(signal) - 1:
            assert i.inst_freq[idx] == expected_inst_freq[idx]
        assert i.inst_cna[idx] == expected_inst_cna[idx]


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


def test_moment_values():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    m = MomentValues(signal)
    assert m.m20 == 0 - 57j
    assert m.m21 == 57
    assert m.m22 == 57j
    assert m.m40 == -6133.200000000001
    assert m.m41 == -6.133200000000001e+03j
    assert m.m42 == 6133.200000000001
    assert m.m43 == 6.133200000000001e+03j
    assert m.m60 == 7.827240000000000e+05j
    assert m.m61 == -782724
    assert m.m62 == -7.827240000000000e+05j
    assert m.m63 == 782724


def calculate_features(features, signal_input):
    result = []
    temp = 0
    for ft in features:
        if ft == 1:
            temp = gmax(signal_input)
        if ft == 2:
            temp = std_dev_abs_inst_phase(signal_input)
        if ft == 3:
            temp = std_dev_inst_phase(signal_input)
        if ft == 4:
            temp = std_dev_abs_inst_cna(signal_input)
        if ft == 5:
            temp = std_dev_inst_cnf(signal_input)
        if ft == 6:
            temp = mean_of_signal_magnitude(signal_input)
        if ft == 7:
            temp = normalized_sqrt_of_sum_of_amp(signal_input)
        if ft == 8:
            temp = kurtosis_of_cn_amplitude(signal_input)
        if ft == 9:
            temp = kurtosis_of_cn_freq(signal_input)
        if ft == 10:
            temp = cumulant_20(signal_input)
        if ft == 11:
            temp = cumulant_21(signal_input)
        if ft == 12:
            temp = cumulant_40(signal_input)
        if ft == 13:
            temp = cumulant_41(signal_input)
        if ft == 14:
            temp = cumulant_42(signal_input)
        if ft == 15:
            temp = cumulant_60(signal_input)
        if ft == 16:
            temp = cumulant_61(signal_input)
        if ft == 17:
            temp = cumulant_62(signal_input)
        if ft == 18:
            temp = cumulant_63(signal_input)
        result.append(temp)
    return result


# 1 - Gmax
def gmax(signal_input):
    signal_abs_fft = np.abs(np.fft.fft(signal_input))
    ft_output = np.max(np.power(signal_abs_fft, 2) / len(signal_input))
    return ft_output


def test_gmax():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert gmax(signal) == 4.0500000000000006e+02


# 2 - Std of the Absolute Instantaneous Phase
def std_dev_abs_inst_phase(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(np.abs(i.inst_phase), ddof=1)
    return ft_output


def test_std_dev_abs_inst_phase():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert std_dev_abs_inst_phase(signal) == 0.940293603578649


# 3 - Std of the Direct Instantaneous Phase
def std_dev_inst_phase(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(i.inst_phase, ddof=1)
    return ft_output


def test_std_dev_inst_phase():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert std_dev_inst_phase(signal) == 1.5903100728408748


# 4 - Std of the CN Instantaneous Amplitude
def std_dev_abs_inst_cna(signal_input):
    i = InstValues(signal_input)
    a = np.abs(i.inst_cna)
    ft_output = np.std(np.abs(i.inst_cna), ddof=1)
    return ft_output


def test_std_dev_abs_inst_cna():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert std_dev_abs_inst_cna(signal) == 0.3312693299999689


# 5 - Std of the Instantaneous Frequency
def std_dev_inst_cnf(signal_input):
    i = InstValues(signal_input)
    ft_output = np.std(i.inst_freq, ddof=1)
    return ft_output


def test_std_dev_abs_inst_cnf():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert std_dev_inst_cnf(signal) == 0.5153882032022075


# 6 - Mean Value of the Signal Magnitude
def mean_of_signal_magnitude(signal_input):
    ft_output = np.mean(np.abs(signal_input))
    return ft_output


def test_mean_of_signal_magnitude():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert mean_of_signal_magnitude(signal) == 6.363961030678928


# 7 - Normalized square root value of sum of amplitude of signal samples
def normalized_sqrt_of_sum_of_amp(signal_input):
    ft_output = np.sqrt(np.sum(np.abs(signal_input))) / len(signal_input)
    return ft_output


def test_normalized_sqrt_of_sum_of_amp():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert normalized_sqrt_of_sum_of_amp(signal) == 0.7977443845417482


# 8 - Kurtosis of the CN Amplitude
def kurtosis_of_cn_amplitude(signal_input):
    i = InstValues(signal_input)
    ft_output = stats.kurtosis(i.inst_cna, fisher=False)
    return ft_output


def test_kurtosis_of_cn_amplitude():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert kurtosis_of_cn_amplitude(signal) == 1.7757575757575754


# 9 - Kurtosis of the CN Frequency
def kurtosis_of_cn_freq(signal_input):
    i = InstValues(signal_input)
    ft_output = stats.kurtosis(i.inst_freq, fisher=False)
    return ft_output


def test_kurtosis_of_cn_freq():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert kurtosis_of_cn_freq(signal) == 1.0627162629757787


def cumulant_20(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m20)


def test_cumulant_20():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_20(signal) == 57


def cumulant_21(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m21)


def test_cumulant_21():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_21(signal) == 57


def cumulant_40(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m40 - 3 * m.m20 * m.m20)


def test_cumulant_40():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_40(signal) == 3613.7999999999993


def cumulant_41(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m41 - 3 * m.m20 * m.m21)


def test_cumulant_41():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_41(signal) == 3613.7999999999993


def cumulant_42(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m42 - np.power(np.abs(m.m20), 2) - 2 * np.power(m.m21, 2))


def test_cumulant_42():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_42(signal) == 3613.7999999999993


def cumulant_60(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m60 - 15 * m.m20 * m.m40 + 3 * np.power(m.m20, 3))


def test_cumulant_60():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_60(signal) == 3905583.000000001


def cumulant_61(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m61 - 5 * m.m21 * m.m40 - 10 * m.m20 * m.m41 + 30 * np.power(m.m20, 2) * m.m21)


def test_cumulant_61():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_61(signal) == 1094627.999999999


def cumulant_62(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m62 - 6 * m.m20 * m.m42 - 8 * m.m21 * m.m41 - m.m22 * m.m40 + \
                  6 * np.power(m.m20, 2) * m.m22 + 24 * np.power(m.m21, 2) * m.m20)


def test_cumulant_62():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_62(signal) == 1094627.999999999


def cumulant_63(signal_input):
    m = MomentValues(signal_input)
    return np.abs(m.m63 - 9 * m.m21 * m.m42 + 12 * np.power(m.m21, 3) - 3 * m.m20 * m.m43 - \
                  3 * m.m22 * m.m41 + 18 * m.m20 * m.m21 * m.m22)


def test_cumulant_63():
    signal = np.array([0 + 0j, -1 + 1j, 2 - 2j, -3 + 3j, 4 - 4j, -5 + 5j, 6 - 6j, -7 + 7j, 8 - 8j, -9 + 9j])
    assert cumulant_63(signal) == 1094627.999999999
