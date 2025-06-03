import random
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


def normal_distribution() -> float:
    u1 = random.random()
    u2 = random.random()
    return np.sqrt(-2 * np.log(u2)) * np.sin(2 * np.pi * u1)


def xi_distribution(n: int) -> float:
    result = []
    for _ in range(n):
        result.append(normal_distribution() ** 2)
    return sum(result)


def show_density_distribution_function(arr: []) -> None:
    bins_num = 3
    if len(arr) > 10:
        bins_num = 20
    plt.hist(arr, bins=bins_num,
             color='hotpink', edgecolor='black', density=True)
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
    print(f"Погрешность для дисперсии {dispersion - 20}")
    print()

    show_density_distribution_function(values)
    show_distribution_function(values)


if __name__ == '__main__':
    count = 10 ** 4
    N = 10

    values = [xi_distribution(N) for _ in range(count)]
    show_info_for_data(values)

