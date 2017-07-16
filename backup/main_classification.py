import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


SAMPLES_TO_REMOVE = 100
X_RANGE = [-3, 2]


def main():
    print "cross validation example with skin dataset"

    err1_arr = np.array(0)
    err2_arr = np.array(0)
    err3_arr = np.array(0)

    for i in range(100):
        err1, err2, err3 = run_example()

        err1_arr = np.append(err1_arr, err1)
        err2_arr = np.append(err2_arr, err2)
        err3_arr = np.append(err3_arr, err3)

    print SAMPLES_TO_REMOVE, " holdout sample score1:", np.mean(err1_arr), " +- ", np.std(err1_arr)
    print "cv10 score1:", np.mean(err2_arr), " +- ", np.std(err2_arr)
    print "entire dataset:", np.mean(err3_arr), " +- ", np.std(err3_arr)


def run_example():
    xy = load_skin_dataset()
    np.random.shuffle(xy)
    x = xy[:, 0:3]
    y = xy[:, [3]]
    y = y.ravel()

    x, y, x_sample, y_sample = remove_random_sample(x, y, SAMPLES_TO_REMOVE)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(np.array(x_sample),
                                                        np.array(y_sample),
                                                        test_size=0.3,
                                                        random_state=0)

    # clf1 = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    solver = "adam"
    ann_max_iter = 1000
    ann = MLPClassifier(solver=solver, max_iter=ann_max_iter, alpha=1e-5, hidden_layer_sizes=(5,), random_state=1).fit(
        x_train, y_train)
    # score1 = calc_error(ann, x_test, y_test)
    #
    # score3 = calc_error(ann, x, y)
    score1 = ann.score(x_test, y_test)
    score2 = np.mean(cross_val_score(ann, x_sample, y_sample, cv=10))
    score3 = ann.score(x, y)

    err1 = 1 - score1
    err2 = 1 - score2
    err3 = 1 - score3

    # print "200 sample score1:", 1 - score1
    # print "cv10 score1:", 1 - score2
    # print "entire dataset:", 1 - score3

    # plt.scatter(x_u, y_u)
    # plt.show()

    return err1, err2, err3


def load_skin_dataset():
    data = pd.read_csv('skinData.txt', delim_whitespace=True, dtype="float64")
    return data.as_matrix()


def remove_random_sample(x, y, sample_n):
    sample = random.sample(range(1, len(x)), sample_n)

    x_u = [x[i] for i in sample]
    y_u = [y[i] for i in sample]

    # x = np.delete(x, sample, 0)
    # y = np.delete(y, sample, 0)

    np.delete(x, sample, 0)
    np.delete(y, sample, 0)

    return x, y, np.array(x_u), np.array(y_u)


def noise():
    return np.random.normal(0, 0.01)

if __name__ == '__main__':
    main()
