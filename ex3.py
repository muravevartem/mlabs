import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging

from prettytable import PrettyTable

AMOUNT_OF_RAND_NUMBER = 100
NUMBER_OF_INTERVALS = 8
TITLE_BAR = 'ex3_bar'


def generator(count):
    res = []
    for i in range(count):
        count_generate = 12
        rand_array = np.random.rand(count_generate)
        res.append(10 + 5 * (sum(rand_array) - 6))
    return res


def getIndexInterval(x, min_x, max_x, h):
    x_start_interval = min_x
    counter = 0
    while x_start_interval <= max_x:
        if x_start_interval <= x <= x_start_interval + h:
            return counter
        if x <= max_x < x_start_interval + 2 * h:
            return counter
        x_start_interval, counter = x_start_interval + h, counter + 1
    raise Exception(f"Illegal argument 'x': {x}")


def intervals(x_array, count_intervals):
    min_x_array = min(x_array)
    max_x_array = max(x_array)
    h = (max_x_array - min_x_array) / count_intervals
    interval_array = [0 for i in range(count_intervals)]
    for x in x_array:
        index = getIndexInterval(x, min_x_array, max_x_array, h)
        logging.info(f"X: {x} INDEX: {index}")
        interval_array[index] += 1
    return min_x_array, max_x_array, h, interval_array


def stat(min_x, max_x, h, n_array, amount_of_rand_number):
    x_array = [min_x + (i * h + h / 2) for i in range(len(n_array))]
    x_v = sum([x_array[i] * n_array[i] for i in range(len(n_array))]) / amount_of_rand_number
    d_v = sum([x_array[i] ** 2 * n_array[i] for i in range(len(n_array))]) / amount_of_rand_number - x_v ** 2
    return x_v, d_v, d_v ** 0.5


def generatorValueOfLaplas(min_x, max_x, h, a, b):
    res = [-0.5]
    a_value = min_x + h
    while a_value <= max_x:
        z = (a_value - a) / b
        res.append(stats.norm.cdf(z) - 0.5)
        a_value += h
    res.append(0.5)
    return res


def stat2(l_array, n_array, amount_of_rand_numbers):
    p_array = []
    for i in range(1, len(l_array)):
        p = l_array[i] - l_array[i - 1]
        p_array.append(p)
        logging.info(f"P: {p}")
    sqr_lambda_v = sum(
        [((n_array[i] - amount_of_rand_numbers * p_array[i]) ** 2) / (amount_of_rand_numbers * p_array[i]) for i in
         range(len(n_array))])
    return p_array, sqr_lambda_v


def drawBar(min_x, max_x, h, amount_of_rand_numbers, n_array):
    y = [n / (amount_of_rand_numbers) * h for n in n_array]
    x = [min_x + h * i for i in range(len(n_array))]
    plt.title(TITLE_BAR)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('nₗ / np')
    plt.bar(x, y, width=h, color='lavender', edgecolor='black', align='edge')
    plt.xticks(np.arange(min_x // 1 - 3, max_x // 1 + 3, 1))
    plt.savefig("ex3_bar.svg")
    plt.show()


def table(min_x, h, number_of_intervals, n_array):
    table = PrettyTable(["a[i-1]", "a[i]", "n[i]"])
    for i in range(number_of_intervals):
        table.add_row([min_x + h * i, min_x + h * (i + 1), n_array[i]])
    print(table)


def main():
    logging.basicConfig(level=logging.INFO)
    x_array = generator(AMOUNT_OF_RAND_NUMBER)
    min_x, max_x, h, n_array = intervals(x_array, NUMBER_OF_INTERVALS)
    logging.info(f"Xmax: {max_x} Xmin: {min_x} h: {h}")
    x_v, d_v, sqrt_d_v = stat(min_x, max_x, h, n_array, AMOUNT_OF_RAND_NUMBER)
    logging.info(f"Хв: {x_v} Дв: {d_v} δ: {sqrt_d_v}")
    res = generatorValueOfLaplas(min_x, max_x, h, x_v, sqrt_d_v)
    logging.info(f"Ф: {res}")
    p_array, sqr_lambda_v = stat2(res, n_array, AMOUNT_OF_RAND_NUMBER)
    logging.info(f"SQR_LAMBDA: {sqr_lambda_v}")
    drawBar(min_x, max_x, h, AMOUNT_OF_RAND_NUMBER, n_array)
    table(min_x, h, NUMBER_OF_INTERVALS, n_array)


main()
