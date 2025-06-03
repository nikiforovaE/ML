import matplotlib.pyplot as plt
import numpy as np
import random
from statsmodels.distributions.empirical_distribution import ECDF


def get_info_for_n(n: int) -> dict:
    arr = [random.random() for _ in range(n)]
    mat_og = sum(arr) / n
    dispersion = sum([(value - mat_og) ** 2 for value in arr]) / n
    sred_kvadr = np.sqrt(dispersion)
    korrelyation = []

    for f in range(1, n + 1):
        up_value = sum((arr[j - 1] - mat_og) * (arr[j + f - 1] - mat_og) for j in range(1, n - f + 1))
        down_value = sum((arr[i - 1] - mat_og) ** 2 for i in range(1, n + 1))
        korrelyation.append(up_value / down_value)

    return {'Математическое ожидание': mat_og,
            'Эмпирическая дисперсия': dispersion,
            'Эмпирическое среднее квадратическое': sred_kvadr,
            'Коэффициенты корреляции': korrelyation,
            'Значения массива': arr}


def show_korrelation_diagram(arr: []) -> None:
    plt.bar(range(len(arr)), arr, color='c')
    plt.xlabel("f")
    plt.ylabel("K(f)")
    plt.xticks([])
    plt.grid(True)
    plt.show()


def show_density_distribution_function(arr: []) -> None:
    bins_num = 3
    if len(arr) > 10:
        bins_num = 20
    plt.hist(arr, bins=bins_num,
             color='violet', edgecolor='black', density=True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()


def show_distribution_function(arr: []) -> None:
    ecdf = ECDF(arr)
    plt.step(ecdf.x, ecdf.y, color="hotpink")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True)
    plt.show()


def show_table(arr: []):
    col_names = ["n", "M", "M (Theoretical value)", "M (Delta)", "D", " D (Theoretical value)", "D (Delta)"]
    table = plt.table(colLabels=col_names, cellText=arr, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    plt.axis('off')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    arr_of_nums = [10, 100, 1000, 10000]
    table = []
    for n in arr_of_nums:
        result = get_info_for_n(n)
        values = result.get('Значения массива')
        mat_og = result.get('Математическое ожидание')
        dispersion = result.get('Эмпирическая дисперсия')
        table.append([n, mat_og, 0.5, mat_og - 0.5, dispersion, 0.08333, dispersion - 0.08333])

        show_korrelation_diagram(result.get('Коэффициенты корреляции'))
        show_distribution_function(values)
        show_density_distribution_function(values)

    show_table(table)
