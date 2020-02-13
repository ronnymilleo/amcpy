import functions


def calculate_features(input_signal):
    f1 = functions.std_deviation((abs(functions.instantaneous_phase(input_signal))))
    f2 = functions.std_deviation(functions.instantaneous_phase(input_signal))
    f3 = functions.std_deviation((abs(functions.instantaneous_frequency(input_signal))))
    f4 = functions.std_deviation(functions.instantaneous_frequency(input_signal))
    f5 = functions.kurtosis(functions.instantaneous_absolute(input_signal))
    f6 = functions.kurtosis(functions.instantaneous_frequency(input_signal))
    f7 = functions.gmax(input_signal)
    f8 = functions.mean_of_squared(functions.instantaneous_cn_absolute(input_signal))
    f9 = functions.std_deviation(abs(functions.instantaneous_cn_absolute(input_signal)))
    f10 = functions.std_deviation(functions.instantaneous_cn_absolute(input_signal))
    f11 = functions.cum20(input_signal)
    f12 = functions.cum21(input_signal)
    f13 = functions.cum40(input_signal)
    f14 = functions.cum41(input_signal)
    f15 = functions.cum42(input_signal)
    f16 = functions.cum60(input_signal)
    f17 = functions.cum61(input_signal)
    f18 = functions.cum62(input_signal)
    f19 = functions.cum63(input_signal)
    f20 = functions.meanAbsolute(input_signal)
    f21 = functions.sqrtAmplitude(input_signal)
    f22 = functions.ratioIQ(input_signal)
    result = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22]
    return result