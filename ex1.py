import numpy as np
from prettytable import PrettyTable
import math
import logging

AMOUNT_OF_RAND_NUMBER = 100
AMOUNT_OF_P = 10
NUMBER_OF_INTERVALS = 8
TITLE_BAR = 'ex2_bar'
TITLE_GRAPH_F = 'F(x)'
TITLE_GRAPH_FX = 'F*(x)'


def combination(n, m):
    return math.factorial(n) / (math.factorial(m) * math.factorial(n - m))


def generator(amount_of_rand_number):
    return np.random.rand(amount_of_rand_number)


def generatorP(amount_of_p):
    p_array = []
    n = amount_of_p
    p = 0.7
    q = 0.3
    for m in range(1, amount_of_p + 1):
        p_el = combination(n, m) * p ** m * q ** (n - m)
        p_array.append(p_el)
    logging.info(f"P sum: {sum(p_array)}")
    return p_array


def mappingInterval(p_array, x_array):
    p_sum, i, i_array = 0, 0, [0]
    for p in p_array:
        p_sum += p
        i_array.append(p_sum)
    i_array.append(1)
    logging.info(f"Intervals: {i_array}")
    w_array = [0 for i in range(len(i_array) - 1)]
    for x in x_array:
        index_interval = -1
        for i in range(len(i_array)):
            if i_array[i] > x:
                index_interval = i - 1
                break
        w_array[index_interval] += 1
        logging.info(f"X: {x} Interval: {i_array[index_interval]} ----> {i_array[index_interval + 1]}")
    return i_array, w_array


def test(w_array, p_array, n):
    s = 0.
    p = 0.7
    q = 0.3
    for i in range(0, len(w_array)):
        npi = n * combination(len(w_array), i + 1) * p ** (i + 1) * q ** (len(w_array) - i - 1)
        s += (w_array[i] - npi) ** 2 / npi
    return s


def stat(w_array, e_array):
    x_v = sum([(i + 1) * w_array[i] for i in range(len(w_array))]) / 100
    d_v = sum([((i + 1) - x_v) ** 2 * w_array[i] for i in range(len(w_array))]) / 100
    return x_v, d_v, d_v ** 0.5


def table(e_array, w_array):
    table = PrettyTable(["Ei", "Wi"])
    for i in range(len(w_array)):
        table.add_row([e_array[i], w_array[i]])
    print(table)


def main():
    logging.basicConfig(level=logging.INFO)
    x_array = generator(AMOUNT_OF_RAND_NUMBER)
    p_array = generatorP(AMOUNT_OF_P)
    i_array, w_array = mappingInterval(p_array, x_array)
    logging.info(f"W: {w_array}")
    e_array = np.arange(1, len(w_array) + 1)
    x_v, d_v, sqrt_d_v = stat(w_array, e_array)
    logging.info(f"Хв: {x_v} Дв: {d_v} δ: {sqrt_d_v}")
    sqr_x = test(w_array, p_array, AMOUNT_OF_RAND_NUMBER)
    logging.info(f"X: {math.sqrt(sqr_x)}")
    table(e_array, w_array)


main()
