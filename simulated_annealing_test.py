import numpy as np
import matplotlib.pyplot as plt
import math


def my_func(x):
    y = x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)
    return y


def judge(delta_E, Tem):
    if delta_E < 0:
        return True
    else:
        d = math.exp(-delta_E / Tem)
        if d > np.random.rand():
            return True
        else:
            return False


def image_plot(var_old, res_old):
    x_array = np.linspace(0, 9, 1000)
    y_array = my_func(x_array)
    x_list = list(x_array)
    y_list = list(y_array)
    max_y = max(y_list)
    max_index = y_list.index(max_y)
    max_x = x_list[max_index]

    plt.plot(x_array, y_array, 'k--', label='my_func')
    plt.plot(max_x, max_y, color='r', marker='x', markersize=10, label='true maximum')
    plt.plot(var_old, res_old, color='b', marker='o', label='result')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simulated Annealing')
    plt.grid()
    plt.savefig('Simulated Annealing Test')
    plt.show()


if __name__ == "__main__":

    lower_bound = 0
    upper_bound = 9
    T = 1e5
    min_T = 1e-3
    alpha = 0.95
    var_old = (upper_bound - lower_bound) * np.random.rand() + lower_bound
    var_new = var_old
    res_old = my_func(var_old)
    res_new = res_old
    counter = 0

    while T > min_T and counter < 10000:
        delta = 3 * (np.random.rand() - 0.5)
        var_new = var_old + delta

        if var_new > upper_bound or var_new < lower_bound:
            var_new = var_new - 2 * delta

        res_new = my_func(var_new)
        dE = res_old - res_new

        if judge(delta_E=dE, Tem=T):
            var_old = var_new
            res_old = res_new

        if delta < 0:
            T = alpha * T
        else:
            counter += 1

    print("old state: ({0}, {1})".format(var_old, res_old))

    image_plot(var_old, res_old)