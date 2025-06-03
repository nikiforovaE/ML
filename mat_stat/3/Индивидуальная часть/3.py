import random

import numpy as np
from matplotlib import pyplot as plt


def triangular_alg(n, a, b):
    # Генерируем случайные числа равномерно распределенные на отрезке [0, 1]
    u = np.random.rand(n)

    # Масштабируем u к интервалу [a, b] для заданных параметров a и b
    scaled_u = (b - a) * u + a

    # Вычисляем значения случайной величины X в соответствии с треугольным распределением
    X = np.where(scaled_u <= 1, np.sqrt(scaled_u), b - np.sqrt(b - scaled_u))

    return X


def normal_arg(n, mean, std_dev):
    # Генерируем две выборки равномерно распределенных на [0, 1]
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)

    # Используем преобразование Бокса-Мюллера для генерации значений с нормальным распределением
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    Y = mean + std_dev * z0

    return Y


def show_density_distribution_function(arr: []) -> None:
    bins_num = 3
    if len(arr) > 10:
        bins_num = 20
    plt.hist(arr, bins=bins_num,
             color='violet', edgecolor='black', density=True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


def sign_test(X, Y, alpha=0.05):
    differences = X - Y
    positive_count = np.sum(differences > 0)
    negative_count = np.sum(differences < 0)

    n = len(differences)

    # Рассчитываем p-value
    p_value = min(positive_count, negative_count) / n

    # Проводим тест
    if p_value < alpha:
        print("Гипотеза H0 отвергается. Распределения неоднородны.")
    else:
        print("Гипотеза H0 не отвергается. Распределения однородны.")
    print("p-value:", p_value)


if __name__ == '__main__':
    count = 50

    a = 0
    b = 2
    m = 1
    std_dev = 1

    values_X = triangular_alg(count, a, b)
    values_Y = normal_arg(count, m, std_dev)
    show_density_distribution_function(values_X)
    show_density_distribution_function(values_Y)

    # Проведем тест
    sign_test(values_X, values_Y)
