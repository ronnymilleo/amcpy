import functions


def calculate_features(input_signal):
    f1 = functions.std_deviation((abs(functions.instantaneous_phase(input_signal))))
    f2 = functions.std_deviation(functions.instantaneous_phase(input_signal))
    f3 = functions.std_deviation((abs(functions.instantaneous_frequency(input_signal))))
    f4 = functions.std_deviation(functions.instantaneous_frequency(input_signal))
    f5 = functions.kurtosis(functions.instantaneous_absolute(input_signal))
    f6 = functions.gmax(input_signal)
    f7 = functions.mean_of_squared(functions.instantaneous_cn_absolute(input_signal))
    f8 = functions.std_deviation(abs(functions.instantaneous_cn_absolute(input_signal)))
    f9 = functions.std_deviation(functions.instantaneous_cn_absolute(input_signal))
    result = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    return result
