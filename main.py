import numpy as np
import sys
from SVM_Optimizer import SVM_Optimizer

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import time

N_SAMPLES = 100000

samples_arr = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def get_seed():
    t = time.time() - int(time.time())
    t *= 10000


def main():
    print "cross validation example with artificial dataset"

    x_all, y_all = make_classification(n_samples=N_SAMPLES, n_features=10, n_redundant=0)

    with open('results/diffAcc.dat', 'a') as real_file:
        real_file.write("#holdout_n \t #diffCV \t #diffCVstd \t #diffholdout \t #diffHoldoutStd \n")

    for n_samples in [50, 100, 150, 200, 250, 300]:
        diff_cv_arr = list()
        diff_holdout_arr = list()

        for i in range(11):
            diff_cv, diff_holdout = run_with_nsample(x_all, y_all, n_samples)
            diff_cv_arr.append(diff_cv)
            diff_holdout_arr.append(diff_holdout)

        with open('results/diffAcc.dat', 'a') as real_file:
            real_file.write(str(n_samples) +\
                            "\t" + str(np.mean(diff_cv_arr)) + "\t" + str(np.std(diff_cv_arr)) +\
                            "\t" + str(np.mean(diff_holdout_arr)) + "\t" + str(np.std(diff_holdout_arr)) +\
                            "\n")


def save_to_file():

    with open('results/cvErr.dat', 'a') as cv_file:
        for x, y, z in zip(cvErrList, cvErrListStd, samples_arr):
            cv_file.write(str(z) + "\t" + str(x) + "\t" + str(y) + '\n')

    with open('results/holdoutErr.dat', 'a') as holdout_file:
        for x, y, z in zip(holdoutErrList, holdoutErrListStd, samples_arr):
            holdout_file.write(str(z) + "\t" + str(x) + "\t" + str(y) + '\n')

    with open('results/realErr.dat', 'a') as real_file:
        for x, y, z in zip(realErrList, realErrListStd, samples_arr):
            real_file.write(str(z) + "\t" + str(x) + "\t" + str(y) + '\n')


def run_with_nsample(x_all, y_all, holdout_n):
    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []

    # n_runs = 2
    #
    # for i in range(n_runs):

    x_holdout, x_without_holdout, y_holdout, y_without_holdout = train_test_split(x_all, y_all, train_size=holdout_n, random_state=get_seed())

    shuffle(x_holdout, y_holdout, random_state=get_seed())
    x_train, x_test, y_train, y_test = train_test_split(x_holdout, y_holdout, test_size=0.3, random_state=get_seed())

    svm_c = calc_svm_parameter(x_train, y_train, x_test, y_test)

    acc1, acc2, acc3, acc4, acc5 = run_example(x_all, y_all, x_holdout, y_holdout, x_without_holdout,y_without_holdout, x_train, y_train, x_test, y_test, svm_c)

    scores1 = np.append(scores1, acc1)
    scores2 = np.append(scores2, acc2)
    scores3 = np.append(scores3, acc3)
    scores4 = np.append(scores4, acc4)
    scores5 = np.append(scores5, acc5)

    # print 'progress:', ((i + 1) / float(n_runs)) * 100, '%'
    #
    # print_result(scores1, scores2, scores3, scores4, scores5, holdout_n)

    print holdout_n, " holdout test sample:", acc1
    print "cv10 on holdout set:", acc2
    print "entire dataset without holdout:", acc3
    print "train dataset:", acc4
    print "entire dataset", acc5

    return abs(acc3 - acc2), abs(acc3 - acc1)


def print_result(scores1, scores2, scores3, scores4, scores5, holdout_n):
    print "\n\nscores:\n"

    print holdout_n, " holdout test sample:", np.mean(scores1), " +- ", np.std(scores1)
    print "cv10 on holdout set:", np.mean(scores2), " +- ", np.std(scores2)
    print "entire dataset without holdout:", np.mean(scores3), " +- ", np.std(scores3)
    print "entire dataset", np.mean(scores5), " +- ", np.std(scores5)
    print "train dataset:", np.mean(scores4), " +- ", np.std(scores4)

    cvErrList.append(np.mean(scores2))
    holdoutErrList.append(np.mean(scores1))
    realErrList.append(np.mean(scores3))

    cvErrListStd.append(np.std(scores2))
    holdoutErrListStd.append(np.std(scores1))
    realErrListStd.append(np.std(scores3))


def calc_svm_parameter(x_train, y_train, x_test, y_test):
    optimizer = SVM_Optimizer(x_train, y_train, x_test, y_test)

    svm_c = optimizer.optimize()
    # svm_c = 0.373874872174 # N_HOLDOUT = 2000
    # svm_c = 0.0761514827158  # N_HOLDOUT = 5000

    print svm_c

    return svm_c



from sklearn.neural_network import MLPClassifier

def run_example(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, x_train, y_train, x_test, y_test, svm_c):

    classifier = svm.SVC(kernel='linear', C=svm_c).fit(x_train, y_train)
    classifierCV = svm.SVC(kernel='linear', C=svm_c)

    # classifier = MLPClassifier(solver='adam',
    #               max_iter=1000,
    #               alpha=0.01,
    #               hidden_layer_sizes=(5,),
    #               random_state=1,
    #               learning_rate='adaptive').fit(x_train, y_train)
    #
    # classifierCV = MLPClassifier(solver='adam',
    #                            max_iter=1000,
    #                            alpha=0.01,
    #                            hidden_layer_sizes=(5,),
    #                            random_state=1,
    #                            learning_rate='adaptive')

    # classifier = DecisionTreeClassifier(max_depth=7).fit(x_train, y_train)
    #
    # classifierCV  = DecisionTreeClassifier(max_depth=7)


    score1 = classifier.score(x_test, y_test) # score on test data set
    scoresCV = cross_val_score(classifierCV, x_holdout, y_holdout, cv=10)
    score2 = np.mean(scoresCV) # estimate accuracy using cv with 5000 samples
    score3 = classifier.score(x_without_holdout, y_without_holdout) # accuracy over entire dataset without holdout samples (5000)
    score4 = classifier.score(x_train, y_train) # accuracy over train samples (should be the highest)
    score5 = classifier.score(x_all, y_all) # accuract over entire dataset

    # print scoresCV
    
    return score1, score2, score3, score4, score5


if __name__ == '__main__':
    main()
