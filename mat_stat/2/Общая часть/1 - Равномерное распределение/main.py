import random
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def uniform_distribution(low_value: float, up_value: float) -> int:
    u = random.random()
    return int(round((up_value - low_value + 1) * u + low_value, 0))


def show_density_distribution_function(arr: []) -> None:
    bins_num = 3
    if len(arr) > 10:
        bins_num = 20
    plt.hist(arr, bins=bins_num,
             color='violet', edgecolor='black', density=True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


def show_distribution_function(arr: []) -> None:
    ecdf = ECDF(arr)
    plt.step(ecdf.x, ecdf.y, color="hotpink")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.show()


if __name__ == '__main__':
    low_value = 1
    up_value = 100
    n = 10 ** 4

    values = [uniform_distribution(low_value, up_value) for _ in range(n)]

    mat_og = sum(values) / n
    dispersion = sum([(value - mat_og) ** 2 for value in values]) / n

    print(f"Мат. ож. {mat_og}")
    print(f"Дисперсия {dispersion}")
    print(f"Погрешность для мат. ож. {mat_og - 50.5}")
    print(f"Погрешность для дисперсии {dispersion - 833.25}")

    show_density_distribution_function(values)
    show_distribution_function(values)
