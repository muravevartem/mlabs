import numpy as np
import matplotlib.pyplot as plt
import logging

from prettytable import PrettyTable

AMOUNT_OF_RAND_NUMBER = 100
NUMBER_OF_INTERVALS = 8
TITLE_BAR = 'ex2_bar'
TITLE_GRAPH_F = 'F(x)'
TITLE_GRAPH_FX = 'F*(x)'


def generator(amount_of_rand_number):
    return np.random.rand(amount_of_rand_number)


def getIndexInterval(x, min_x, max_x, h):
    x_start_interval = min_x
    counter = 0
    while x_start_interval <= max_x:
        if x_start_interval <= x <= x_start_interval + h:
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


def stat(min_x, max_x, h, n_array):
    x_array = [min_x + (i * h + h / 2) for i in range(len(n_array))]
    x_v = sum([x_array[i] * n_array[i] for i in range(len(n_array))]) / 100
    d_v = sum([x_array[i] ** 2 * n_array[i] for i in range(len(n_array))]) / 100 - x_v ** 2
    return x_v, d_v, d_v ** (1 / 2)


def fx(x, min_x, max_x, h, n_array):
    if x <= min_x:
        return 0
    if x > 1 - h:
        return 1
    interval = getIndexInterval(x, min_x, max_x, h) + 1
    return sum([n_array[i] for i in range(interval)]) / 100


def f(x):
    if x <= 0: return 0
    if x >= 1: return 1
    return x * x ** 0.5


def drawBar(min_x, max_x, h, amount_of_rand_numbers, n_array):
    y = [n / (amount_of_rand_numbers) * h for n in n_array]
    x = [min_x + h * i for i in range(len(n_array))]
    plt.title(TITLE_BAR)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('nₗ / np')
    plt.bar(x, y, width=h, color='lavender', edgecolor='black', align='edge')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.savefig("ex2_bar.svg")
    plt.show()


def drawGraph(min_x, max_x, h, n_array):
    plt.title(TITLE_GRAPH_F)
    plt.grid(True)
    x_array = np.arange(-1,2,0.001)
    plt.plot(x_array, [f(x) for x in x_array])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.savefig("ex2_graph_f(x).svg")
    plt.show()
    plt.clf()
    plt.title(TITLE_GRAPH_FX)
    plt.grid(True)
    plt.plot(x_array, [fx(x, min_x, max_x, h, n_array) for x in x_array])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.savefig("ex2_graph_fx(x).svg")
    plt.show()

def test(min_x, max_x, h, n_array):
    d_array = [f(min_x + h * x) - fx(min_x + h * x, min_x, max_x, h, n_array) for x in range(len(n_array))]
    d_n = max(d_array)
    return d_n, d_n * 10


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
    x_v, d_v, sqrt_d_v = stat(min_x, max_x, h, n_array)
    logging.info(f"Хв: {x_v} Дв: {d_v} δ: {sqrt_d_v}")
    d_n, lambda_b = test(min_x, max_x, h, n_array)
    logging.info(f"Dn: {d_n} λв: {lambda_b}")
    drawBar(min_x, max_x, h, AMOUNT_OF_RAND_NUMBER, n_array)
    drawGraph(min_x, max_x, h, n_array)
    table(min_x, h, NUMBER_OF_INTERVALS, n_array)

main()