import numpy as np
import logging

AMOUNT_OF_RAND_NUMBER = 100
NUMBER_OF_INTERVALS = 8


def generator(amount_of_rand_number):
    return [(1 - (1 - u ** 2) ** 0.5) / u for u in np.random.rand(amount_of_rand_number)]


def get_index_interval(x, min_x, max_x, h):
    x_start_interval = min_x
    counter = 0
    while x_start_interval <= max_x:
        if x_start_interval <= x < x_start_interval + h:
            return counter
        if x == max_x:
            return counter
        x_start_interval, counter = x_start_interval + h, counter + 1
    raise Exception(f"Illegal argument 'x': {x}")


def intervals(x_array, count_intervals):
    min_x_array = min(x_array)
    max_x_array = max(x_array)
    h = (max_x_array - min_x_array) / count_intervals
    interval_array = [0 for i in range(count_intervals)]
    for x in x_array:
        index = get_index_interval(x, min_x_array, max_x_array, h)
        logging.info(f"X: {x} INDEX: {index}")
        interval_array[index] += 1
    return min_x_array, max_x_array, h, interval_array


def stat(min_x, max_x, h, n_array, amount_of_rand_number):
    x_array = [min_x + (i * h + h / 2) for i in range(len(n_array))]
    x_v = sum([x_array[i] * n_array[i] for i in range(len(n_array))]) / amount_of_rand_number
    d_v = sum([x_array[i] ** 2 * n_array[i] for i in range(len(n_array))]) / amount_of_rand_number - x_v ** 2
    return x_v, d_v, d_v ** (0.5)


def fx(x, min_x, max_x, h, n_array):
    if x <= min_x:
        return 0
    if x > 1 - h:
        return 1
    interval = get_index_interval(x, min_x, max_x, h) + 1
    return sum([n_array[i] for i in range(interval)]) / AMOUNT_OF_RAND_NUMBER


def f(x):
    if x <= 0: return 0
    if x >= 1: return 1
    return x * x ** 0.5


def test(min_x, max_x, h, n_array):
    d_array = [f(min_x + h * x) - fx(min_x + h * x, min_x, max_x, h, n_array) for x in range(len(n_array))]
    d_n = max(d_array)
    return d_n, d_n * 10


def main():
    logging.basicConfig(level=logging.INFO)
    x_array = generator(AMOUNT_OF_RAND_NUMBER)
    print([x_array[i] for i in range(12)])
    min_x, max_x, h, n_array = intervals(x_array, NUMBER_OF_INTERVALS)
    logging.info(f"Xmax: {max_x} Xmin: {min_x} h: {h}")
    x_v, d_v, sqrt_d_v = stat(min_x, max_x, h, n_array, AMOUNT_OF_RAND_NUMBER)
    logging.info(f"Хв: {x_v} Дв: {d_v} δ: {sqrt_d_v}")
    d_n, lambda_b = test(min_x, max_x, h, n_array)
    logging.info(f"Dn: {d_n} λв: {lambda_b}")


main()
