import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from Utils import Utils

SAMPLES_N = 10000
SAMPLES_TO_REMOVE = 1000

X_RANGE = [-3, 2]


def main():
    print "cross validation example with artificial dataset"

    err1_arr = np.empty(1)
    err2_arr = np.empty(1)
    err3_arr = np.empty(1)
    err4_arr = np.empty(1)
    err5_arr = np.empty(1)

    N_RUNS = 10

    for i in range(N_RUNS):
        err1, err2, err4, err5 = run_example()

        err1_arr = np.append(err1_arr, err1)
        err2_arr = np.append(err2_arr, err2)
        err4_arr = np.append(err4_arr, err4)
        err5_arr = np.append(err5_arr, err5)

        print 'progress:', (i/float(N_RUNS))*100, '%'

    print SAMPLES_TO_REMOVE, " holdout sample score1:", np.mean(err1_arr), " +- ", np.std(err1_arr)
    print "cv10 score1:", np.mean(err2_arr), " +- ", np.std(err2_arr)
    print "entire dataset", np.mean(err5_arr), " +- ", np.std(err5_arr)
    print "train dataset:", np.mean(err4_arr), " +- ", np.std(err4_arr)


def run_example():
    a = 0.01
    b = 0.01
    c = -0.04
    d = 0
    e = 0.08
    f = 0.01

    x, y = generate_random_poly_with_noise(X_RANGE, SAMPLES_N, [a, b, c, d, e, f])

    x_without_holdout, y_without_holdout, x_sample, y_sample = Utils.remove_random_sample(x, y, SAMPLES_TO_REMOVE)
    x_train, x_test, y_train, y_test = train_test_split(np.array(x_sample),
                                                        np.array(y_sample),
                                                        test_size=0.3,
                                                        random_state=0)

    solver = "adam"
    ann_max_iter = 1000

    print x_train.shape
    print y_train.shape

    y_train.reshape(len(x_train), 1)
    x_train.reshape(len(x_train), 1)

    print x_train.shape
    print y_train.shape

    regressor = MLPRegressor(solver=solver, max_iter=ann_max_iter, alpha=1e-5, hidden_layer_sizes=(5,), random_state=1).fit(
        x_train, y_train)

    err1 = regressor.score(x_test, y_test)
    err2 = np.mean(cross_val_score(regressor, x_sample, y_sample, cv=10))
    err4 = regressor.score(x_train, y_train)
    err5 = regressor.score(x_without_holdout, y_without_holdout)

    # plt.scatter(x_u, y_u)
    # plt.show()

    return err1, err2, err4, err5


def random_sample(x, y, sample_n):
    sample = random.sample(range(1, len(x)), sample_n)

    x_u = [x[i] for i in sample]
    y_u = [y[i] for i in sample]

    return x_u, y_u


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
        y[i] = calc_poly(x[i], coefficients)  # + noise()

    return x, y


def calc_error(ann, x, y):
    y_p = ann.predict(x)

    err = 0.0

    for y1, y2 in zip(y, y_p):
        err += (y1-y2)**2

    return err/len(x)


def calc_error_cv(ann, x, y, folds_n):
    errors = np.zeros(folds_n)

    for i in range(folds_n):
        errors[i] = calc_error(ann, x, y)

    return np.mean(errors)


if __name__ == '__main__':
    main()
