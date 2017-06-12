import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

from Utils import Utils

HOLDOUT_SAMPLES = 2000
N_SAMPLES = 20000


def main():
    print "cross validation example with artificial dataset"

    err1_arr = np.empty(1)
    err2_arr = np.empty(1)
    err4_arr = np.empty(1)
    err5_arr = np.empty(1)

    N_RUNS = 10

    x, y = make_classification(n_samples=N_SAMPLES, n_features=5, n_redundant=0)

    for i in range(N_RUNS):
        err1, err2, err4, err5 = run_example(x, y)

        err1_arr = np.append(err1_arr, err1)
        err2_arr = np.append(err2_arr, err2)
        err4_arr = np.append(err4_arr, err4)
        err5_arr = np.append(err5_arr, err5)

        print 'progress:', (i/float(N_RUNS))*100, '%'

    print "\n\nerrors:\n"

    print HOLDOUT_SAMPLES, " holdout test sample:", np.mean(err1_arr), " +- ", np.std(err1_arr)
    print "cv10 score1:", np.mean(err2_arr), " +- ", np.std(err2_arr)
    print "entire dataset", np.mean(err5_arr), " +- ", np.std(err5_arr)
    print "train dataset:", np.mean(err4_arr), " +- ", np.std(err4_arr)


def run_example(x, y):

    x_without_holdout, y_without_holdout, x_sample, y_sample = Utils.remove_random_sample(x, y, HOLDOUT_SAMPLES)
    x_train, x_test, y_train, y_test = train_test_split(np.array(x_sample),
                                                        np.array(y_sample),
                                                        test_size=0.3,
                                                        random_state=0)


    # solver = "adam"
    # ann_max_iter = 1000

    # classifier = MLPClassifier(solver=solver, max_iter=ann_max_iter, alpha=1e-5, hidden_layer_sizes=(5,), random_state=1).fit(x_train, y_train)

    classifier = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

    score1 = classifier.score(x_test, y_test)
    score2 = np.mean(cross_val_score(classifier, x_sample, y_sample, cv=10))
    score4 = classifier.score(x_train, y_train)
    score5 = classifier.score(x, y)

    err1 = 1 - score1
    err2 = 1 - score2
    err4 = 1 - score4
    err5 = 1 - score5

    # plt.scatter(x_u, y_u)
    # plt.show()

    return err1, err2, err4, err5


if __name__ == '__main__':
    main()
