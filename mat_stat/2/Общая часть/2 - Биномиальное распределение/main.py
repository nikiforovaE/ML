import random
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np


def binomial_distribution(n: int, p: float) -> int:
    result_m = 0
    if n < 100:
        value = random.random()
        p_func = (1 - p) ** n
        while value >= p_func:
            value = value - p_func
            p_func = p_func * ((p * (n - result_m)) / ((result_m + 1) * (1 - p)))
            result_m = result_m + 1
    else:
        result_m = round(np.random.normal(loc=(n * p), scale=np.sqrt(n * p * (1 - p))))
    return result_m


def show_density_distribution_function(arr: []) -> None:
    bins_num = 20
    plt.hist(arr, bins=bins_num,
             color='hotpink', edgecolor='black', density=True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


def show_distribution_function(arr: []) -> None:
    ecdf = ECDF(arr)
    plt.step(ecdf.x, ecdf.y, color='violet')
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.show()


if __name__ == '__main__':
    count = 10 ** 4
    N = 10
    probability = 0.5
    values = [binomial_distribution(N, probability) for _ in range(count)]

    mat_og = sum(values) / count
    dispersion = sum([(value - mat_og) ** 2 for value in values]) / count

    print(f"Мат. ож. {mat_og}")
    print(f"Дисперсия {dispersion}")
    print(f"Погрешность для мат. ож. {mat_og - 5}")
    print(f"Погрешность для дисперсии {dispersion - 2.5}")

    show_density_distribution_function(values)
    show_distribution_function(values)
