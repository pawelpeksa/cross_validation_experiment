import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

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

    x = x.reshape(-1, 1)
    y = y.ravel()

    # plt.scatter(x, y)
    # plt.show()

    x_sample, y_sample = random_sample(x, y, 200)

    x_train, x_test, y_train, y_test = train_test_split(np.array(x_sample),
                                                        np.array(y_sample),
                                                        test_size=0.3,
                                                        random_state=0)

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    y_test = y_test.ravel()
    y_train = y_train.ravel()

    # clf1 = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    solver = "adam"
    ann_max_iter = 1000

    ann = MLPRegressor(solver=solver, max_iter=ann_max_iter, alpha=1e-5, hidden_layer_sizes=(5,), random_state=1).fit(
        x_train, y_train)

    score = calc_error(ann, x_test, y_test)

    score2 = calc_error(ann, x, y)

    scores = cross_val_score(ann, x_sample, y_sample, cv=10)

    print "200 sample score:", score
    print "cv10 score:", np.mean(scores)
    print "entire dataset:", score2

    # plt.scatter(x_u, y_u)
    # plt.show()


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
