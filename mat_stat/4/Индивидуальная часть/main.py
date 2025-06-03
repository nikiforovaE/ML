import numpy as np
import math


def logic_func(x):
    t_value = 8760
    return (((x[0] > t_value and x[1] > t_value) or (x[2] > t_value and x[3] > t_value))
            and x[4] > t_value and x[5] > t_value
            and (x[6] > t_value or x[7] > t_value or x[8] > t_value)
            and (x[9] > t_value or x[10] > t_value))


def find_P(L):
    N = 33013
    n = [4, 2, 3, 2]
    m = 4
    lambdas_values = [40 * 10 ** (-6), 10 * 10 ** (-6), 80 * 10 ** (-6), 30 * 10 ** (-6)]
    d_counter = 0

    for k in range(N):
        x = []
        for i in range(m):
            t = []
            for j in range(n[i]):
                alfa = np.random.uniform(0, 1)
                t.append((-math.log(alfa) / lambdas_values[i]))
            for j in range(L[i]):
                l = t.index(min(t))
                t[l] -= math.log(np.random.uniform(0, 1)) / lambdas_values[i]
            for j in range(n[i]):
                x.append(t[j])
        d_counter += not logic_func(x)

    P = 1 - d_counter / N
    return P


if __name__ == '__main__':
    L = [0, 0, 0, 0]
    arr = [4, 2, 3, 2]
    P0 = 0.995
    for i in range(arr[0], arr[0] + 3):
        L[0] = i
        for j in range(arr[1], arr[1] + 3):
            L[1] = j
            for k in range(arr[2], arr[2] + 3):
                L[2] = k
                for l in range(arr[3], arr[3] + 3):
                    L[3] = l
                    P = find_P(L)
                    if P > P0:
                        print(f'L = {L}, sum = {sum(L)}, P = {P}\n')
