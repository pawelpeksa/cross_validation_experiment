import numpy as np
import matplotlib.pyplot as plt
import random

SAMPLES_N = 5000
X_RANGE = [-3, 2]


def main():
    print "cross validation example"

    a = 0.01
    b = 0.01
    c = -0.04
    d = 0
    e = 0.08
    f = 0.01

    x, y = generate_random_poly_with_noise(X_RANGE, SAMPLES_N, [a, b, c, d, e, f])

    # plt.scatter(x, y)
    # plt.show()

    num_to_select = 200  # set the number to select here.
    list_of_random_items = random.sample(x, num_to_select)

    sample = random.sample(range(1, len(x)), num_to_select)

    x_u = [x[i] for i in sample]
    y_u = [y[i] for i in sample]

    print len(x_u), " ", len(y_u)

    # plt.scatter(x_u, y_u)
    # plt.show()


def noise():
    return np.random.normal(0, 0.01)


def calc_poly(x, coefficients):
    result = 0

    i = len(coefficients) - 1

    for coeff in coefficients:
        result += coeff * x ** i
        i -= 1

    return result


def generate_random_poly_with_noise(x_range, samples_n, coefficients):
    x = np.linspace(x_range[0], x_range[1], SAMPLES_N)
    y = np.zeros(samples_n)

    for i in range(0, SAMPLES_N):
        y[i] = calc_poly(x[i], coefficients) # + noise()

    return x, y

if __name__ == '__main__':
    main()
