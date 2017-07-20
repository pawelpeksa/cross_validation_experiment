import numpy as np
import sys
from SVM_Optimizer import SVM_Optimizer

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import time

import matplotlib.pyplot as plt

N_SAMPLES = 100000
N_HOLDOUT = 5000


def main():
    print "cross validation example with artificial dataset"

    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []

    n_runs = 5

    x_all, y_all = make_classification(n_samples=N_SAMPLES, n_features=30, n_redundant=0)
    
    # plt.scatter(x_all[:, 0], x_all[:, 1], marker='o', c=y_all)
    # plt.show()
    
    x_holdout, x_without_holdout, y_holdout, y_without_holdout = train_test_split(x_all, y_all, train_size=N_HOLDOUT, random_state=int(int(time.time())))

    svm_c = calc_svm_parameter(x_holdout, y_holdout)

    for i in range(n_runs):
        err1, err2, err3, err4, err5 = run_example(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, svm_c)

        scores1 = np.append(scores1, err1)
        scores2 = np.append(scores2, err2)
        scores3 = np.append(scores3, err3)
        scores4 = np.append(scores4, err4)
        scores5 = np.append(scores5, err5)

        print 'progress:', ((i+1)/float(n_runs))*100, '%'

    print_result(scores1, scores2, scores3, scores4, scores5)


def print_result(scores1, scores2, scores3, scores4, scores5):
    print "\n\nscores:\n"

    print N_HOLDOUT, " holdout test sample:", np.mean(scores1), " +- ", np.std(scores1)
    print "cv10 on holdout set:", np.mean(scores2), " +- ", np.std(scores2)
    print "entire dataset without holdout:", np.mean(scores3), " +- ", np.std(scores3)
    print "entire dataset", np.mean(scores5), " +- ", np.std(scores5)
    print "train dataset:", np.mean(scores4), " +- ", np.std(scores4)


def calc_svm_parameter(x, y):
    optimizer = SVM_Optimizer(x, y)

    # svm_c = optimizer.optimize()
    # svm_c = 0.373874872174 # N_HOLDOUT = 2000
    svm_c = 0.0761514827158  # N_HOLDOUT = 5000

    print svm_c

    return svm_c


def run_example(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, svm_c):

    x_train, x_test, y_train, y_test = train_test_split(x_holdout, y_holdout, test_size=0.3, random_state=int(int(time.time())))

    classifier = svm.SVC(gamma=1, C=10).fit(x_train, y_train)
    classifierCV = svm.SVC(gamma=1, C=10)
#    classifier = svm.SVC(kernel='linear', C=svm_c).fit(x_train, y_train)
#    classifierCV = svm.SVC(kernel='linear', C=svm_c)

    score1 = classifier.score(x_test, y_test) # score on test data set

    tmp, x_holdout, tmp, y_holdout = train_test_split(x_holdout, y_holdout, test_size=0.9999999, random_state=int(int(time.time())))

    cvscore = cross_val_score(classifierCV, x_holdout, y_holdout, cv=10)
    score2 = np.mean(cvscore)
#    score2 = np.mean(cross_val_score(classifierCV, x_holdout, y_holdout, cv=10)) # estimate accuracy using cv with 5000 samples
    score3 = classifier.score(x_without_holdout, y_without_holdout) # accuracy over entire dataset without holdout samples (5000)
    score4 = classifier.score(x_train, y_train) # accuracy over train samples (should be the highest)
    score5 = classifier.score(x_all, y_all) # accuract over entire dataset

    print "SVM: ",score1
    print "SVM CV: ",score2
    print "SVM CV: ", cvscore

    return score1, score2, score3, score4, score5


if __name__ == '__main__':
    main()
