import random
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def uniform_distribution(a: float, b: float) -> float:
    u = random.random()
    return (b - a) * u + a


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
    plt.step(ecdf.x, ecdf.y, color='violet')
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.show()


if __name__ == '__main__':
    low_value = 1
    up_value = 10
    n = 10 ** 4

    values = [uniform_distribution(low_value, up_value) for _ in range(n)]

    mat_og = sum(values) / n
    dispersion = sum([(value - mat_og) ** 2 for value in values]) / n

    print(f"Мат. ож. {mat_og}")
    print(f"Дисперсия {dispersion}")
    print(f"Теоретическое Мат. ож. {(low_value + up_value) / 2}")
    print(f"Теоретическая дисперсия {((up_value - low_value) ** 2) / 12}")
    print(f"Погрешность для мат. ож. {mat_og - (low_value + up_value) / 2}")
    print(f"Погрешность для дисперсии {dispersion - ((up_value - low_value) ** 2) / 12}")

    show_density_distribution_function(values)
    show_distribution_function(values)
