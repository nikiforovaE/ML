import math
from collections import Counter

import numpy as np
import random

from scipy.stats import geom
from scipy.stats import chisquare


def third_algorithm(p: float) -> int:
    value = random.random()
    result_m = math.floor(np.log(value) / np.log(1 - p)) + 1
    return result_m


if __name__ == '__main__':
    count = 50
    p = 0.7
    n = 50

    values = [third_algorithm(p) for _ in range(n)]

    frequency = Counter(values)
    observed_freq, _ = np.histogram(values, bins=np.arange(1, np.max(values) + 2))
    observed_freq_sum = np.sum(observed_freq)

    expected_freq = geom.pmf(np.arange(1, np.max(values) + 1), p) * n
    expected_freq_sum = np.sum(expected_freq)

    observed_freq_norm = observed_freq / observed_freq_sum
    expected_freq_norm = expected_freq / expected_freq_sum
    chi2_stat, p_value = chisquare(observed_freq_norm, expected_freq_norm)

    print("Хи-квадрат статистика:", chi2_stat)
    print("p-value:", p_value)

    if p_value < 0.05:
        print("Отвергаем нулевую гипотезу о том, что эмпирическое распределение согласуется с теоретическим")
    else:
        print("Не отвергаем нулевую гипотезу о том, что эмпирическое распределение согласуется с теоретическим")
