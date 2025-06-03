import math
import random
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np


def first_algorithm(p: float) -> int:
    value = random.random()
    p_func = p
    result_m = 0
    while value >= p_func:
        value = value - p_func
        p_func = p_func * (1 - p)
        result_m = result_m + 1
    return result_m + 1


def second_algorithm(p: float) -> int:
    value = random.random()
    result_m = 0
    p_func = p
    while value >= p_func:
        value = random.random()
        result_m = result_m + 1
    return result_m + 1


def third_algorithm(p: float) -> int:
    value = random.random()
    result_m = math.floor(np.log(value) / np.log(1 - p)) + 1
    return result_m


def show_density_distribution_function(arr: []) -> None:
    bins_num = 13
    plt.hist(arr, bins=bins_num,
             color='violet', edgecolor='black', density=True)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


def show_distribution_function(arr: []) -> None:
    ecdf = ECDF(arr)
    plt.step(ecdf.x, ecdf.y)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.show()


def show_info_for_data(values: []) -> None:
    mat_og = sum(values) / count
    dispersion = sum([(value - mat_og) ** 2 for value in values]) / count

    print(f"Мат. ож. {mat_og}")
    print(f"Дисперсия {dispersion}")
    print(f"Погрешность для мат. ож. {mat_og - 2}")
    print(f"Погрешность для дисперсии {dispersion - 2.2}")
    print()

    show_density_distribution_function(values)
    show_distribution_function(values)


if __name__ == '__main__':
    count = 10 ** 4
    probability = 0.5

    values = [first_algorithm(probability) for _ in range(count)]
    show_info_for_data(values)

    values = [second_algorithm(probability) for _ in range(count)]
    show_info_for_data(values)

    values = [third_algorithm(probability) for _ in range(count)]
    show_info_for_data(values)
