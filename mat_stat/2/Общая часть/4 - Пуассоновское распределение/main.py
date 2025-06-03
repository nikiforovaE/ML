import random
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np


def first_algorithm(mu: float) -> int:
    result_m = 0
    if mu < 88:
        value = random.random()
        p_func = np.exp(-mu)
        result_m = 1
        while value >= p_func:
            value = value - p_func
            p_func = p_func * (mu / result_m)
            result_m = result_m + 1
    else:
        result_m = round(np.random.normal(1, mu, mu))
    return result_m - 1


def second_algorithm(mu: float) -> int:
    result_m = 0
    if mu < 88:
        result_m = 1
        p_func = random.random()
        exp_value = np.exp(-mu)
        while p_func >= exp_value:
            value = random.random()
            p_func = p_func * value
            result_m = result_m + 1
    else:
        result_m = round(np.random.normal(1, mu, mu))
    return result_m - 1


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
    print(f"Погрешность для мат. ож. {mat_og - 10}")
    print(f"Погрешность для дисперсии {dispersion - 10}")
    print()

    show_density_distribution_function(values)
    show_distribution_function(values)


if __name__ == '__main__':
    count = 10 ** 4
    mu = 10

    values = [first_algorithm(mu) for _ in range(count)]
    show_info_for_data(values)

    values = [second_algorithm(mu) for _ in range(count)]
    show_info_for_data(values)
