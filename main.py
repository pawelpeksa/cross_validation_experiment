from SVM_Optimizer import SVM_Optimizer
from ANN_Optimizer import ANN_Optimizer
from DecisionTree_Optimizer import DecisionTree_Optimizer
from RandomForest_Optimizer import RandomForest_Optimizer
from MethodsConfiguration import MethodsConfiguration

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle

from copy import deepcopy
import time
import numpy as np

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
            diff_cv, diff_holdout = optimize_and_score(x_all, y_all, n_samples)
            diff_cv_arr.append(diff_cv)
            diff_holdout_arr.append(diff_holdout)

        with open('results/diffAcc.dat', 'a') as real_file:
            real_file.write(str(n_samples) +\
                            "\t" + str(np.mean(diff_cv_arr)) + "\t" + str(np.std(diff_cv_arr)) +\
                            "\t" + str(np.mean(diff_holdout_arr)) + "\t" + str(np.std(diff_holdout_arr)) +\
                            "\n")


def optimize_and_score(x_all, y_all, holdout_n):
    x_holdout, x_without_holdout, y_holdout, y_without_holdout = train_test_split(x_all, y_all, train_size=holdout_n, random_state=get_seed())

    shuffle(x_holdout, y_holdout, random_state=get_seed())
    x_train, x_test, y_train, y_test = train_test_split(x_holdout, y_holdout, test_size=0.3, random_state=get_seed())

    config = determine_parameters_all(x_train, y_train, x_test, y_test)

    clf = svm.SVC(kernel='linear', C=config.svm.C)
    acc1, acc2, acc3, acc4, acc5 = score_model(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, x_train, y_train, x_test, y_test, clf)

    print holdout_n, " holdout test sample:", acc1
    print "cv10 on holdout set:", acc2
    print "entire dataset without holdout:", acc3
    print "train dataset:", acc4
    print "entire dataset", acc5

    return abs(acc3 - acc2), abs(acc3 - acc1)


def determine_parameters_all(x_train, y_train, x_test, y_test):
    print "determine parameters"
    config = MethodsConfiguration()

    config.svm.C = determine_parameters(SVM_Optimizer(x_train, y_train, x_test, y_test))
    # config.ann.hidden_neurons, config.ann.solver, config.ann.alpha = determine_parameters(ANN_Optimizer(x_train, y_train, x_test, y_test))
    # config.decision_tree.max_depth = determine_parameters(DecisionTree_Optimizer(x_train, y_train, x_test, y_test))
    # config.random_forest.max_depth, config.random_forest.n_estimators = determine_parameters(RandomForest_Optimizer(x_train, y_train, x_test, y_test))

    return config


def determine_parameters(optimizer):
    print 'determine parameters', optimizer.__class__.__name__
    return optimizer.optimize()


def score_model(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, x_train, y_train, x_test, y_test, classifier):
    classifierCV = deepcopy(classifier)

    classifier.fit(x_train, y_train)

    score1 = classifier.score(x_test, y_test) # score on test data set
    scoresCV = cross_val_score(classifierCV, x_holdout, y_holdout, cv=10)
    score2 = np.mean(scoresCV) # estimate accuracy using cv with 5000 samples
    score3 = classifier.score(x_without_holdout, y_without_holdout) # accuracy over entire dataset without holdout samples (5000)
    score4 = classifier.score(x_train, y_train) # accuracy over train samples (should be the highest)
    score5 = classifier.score(x_all, y_all) # accuract over entire dataset

    return score1, score2, score3, score4, score5


if __name__ == '__main__':
    main()
