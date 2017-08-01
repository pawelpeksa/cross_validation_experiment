from copy import deepcopy
import time
import threading
import numpy as np

from MethodsConfiguration import MethodsConfiguration
from Optimizer import *

from Configuration import Configuration

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle

TREE_KEY = 'tree'
FOREST_KEY = 'forest'
ANN_KEY = 'ann'
SVM_KEY = 'svm'


def get_seed():
    t = time.time() - int(time.time())
    t *= 1000000
    return int(t)


def main():
    print "cross validation example with artificial dataset"

    x_all, y_all = make_classification(n_samples=Configuration.N_SAMPLES, n_features=10, n_redundant=0)

    open_file_with_header(SVM_KEY)
    open_file_with_header(ANN_KEY)
    open_file_with_header(TREE_KEY)
    open_file_with_header(FOREST_KEY)

    for n_samples in Configuration.n_samples_arr:
        result_dict = dict()

        result_dict[SVM_KEY] = list(), list()
        result_dict[ANN_KEY] = list(), list()
        result_dict[TREE_KEY] = list(), list()
        result_dict[FOREST_KEY] = list(), list()

        for i in range(Configuration.RUNS_FOR_SAMPLE + 1):
            single_result_dict = optimize_and_score(x_all, y_all, n_samples)

            append_to_result_array(single_result_dict, result_dict, SVM_KEY)
            append_to_result_array(single_result_dict, result_dict, ANN_KEY)
            append_to_result_array(single_result_dict, result_dict, TREE_KEY)
            append_to_result_array(single_result_dict, result_dict, FOREST_KEY)

        append_result_to_file(SVM_KEY, n_samples, *(result_dict[SVM_KEY]))
        append_result_to_file(ANN_KEY, n_samples, *(result_dict[ANN_KEY]))
        append_result_to_file(TREE_KEY, n_samples, *(result_dict[TREE_KEY]))
        append_result_to_file(FOREST_KEY, n_samples, *(result_dict[FOREST_KEY]))


def append_to_result_array(single_result_dict, result_dict, KEY):
    diff_cv_arr, diff_holdout_arr = result_dict[KEY]
    diff_cv, diff_holdout = single_result_dict[KEY]
    diff_cv_arr.append(diff_cv)
    diff_holdout_arr.append(diff_holdout)


def append_result_to_file(name, n_samples, diff_cv_arr, diff_holdout_arr):
    with open('results/' + name + '.dat', 'a') as file:
        file.write(str(n_samples) + \
                   "\t" + str(np.mean(diff_cv_arr)) + "\t" + str(np.std(diff_cv_arr)) + \
                   "\t" + str(np.mean(diff_holdout_arr)) + "\t" + str(np.std(diff_holdout_arr)) + \
                   "\n")


def open_file_with_header(name):
    with open('results/' + name + '.dat', 'a') as file:
        file.write("#holdout_n \t #diffCV \t #diffCVstd \t #diffholdout \t #diffHoldoutStd \n")


def optimize_and_score(x_all, y_all, holdout_n):
    x_holdout, x_without_holdout, y_holdout, y_without_holdout = train_test_split(x_all, y_all, train_size=holdout_n,
                                                                                  random_state=get_seed())

    shuffle(x_holdout, y_holdout, random_state=get_seed())
    x_train, x_test, y_train, y_test = train_test_split(x_holdout, y_holdout, test_size=0.3, random_state=get_seed())

    config = determine_parameters_all(x_train, y_train, x_test, y_test)

    SVM = svm.SVC(kernel='linear', C=config.svm.C)

    ann = MLPClassifier(solver=config.ann.solver,
                        max_iter=Configuration.ANN_MAX_ITERATIONS,
                        alpha=config.ann.alpha,
                        hidden_layer_sizes=(config.ann.hidden_neurons,),
                        learning_rate='adaptive')

    tree = DecisionTreeClassifier(max_depth=config.decision_tree.max_depth)

    forest = RandomForestClassifier(max_depth=config.random_forest.max_depth,
                                    n_estimators=config.random_forest.n_estimators)

    score_dict = dict()

    score_dict[SVM_KEY] = score_model(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, x_train,
                                      y_train, x_test, y_test, SVM)
    score_dict[ANN_KEY] = score_model(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, x_train,
                                      y_train, x_test, y_test, ann)
    score_dict[FOREST_KEY] = score_model(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout,
                                         x_train, y_train, x_test, y_test, tree)
    score_dict[TREE_KEY] = score_model(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout,
                                       x_train, y_train, x_test, y_test, forest)

    return score_dict


def determine_parameters_all(x_train, y_train, x_test, y_test):
    print "determine parameters"
    config = MethodsConfiguration()

    print config.toDict()

    threads = list()

    svm_opt = SVM_Optimizer(x_train, y_train, x_test, y_test)
    ann_opt = ANN_Optimizer(x_train, y_train, x_test, y_test)
    tree_opt = DecisionTree_Optimizer(x_train, y_train, x_test, y_test)
    forest_opt = RandomForest_Optimizer(x_train, y_train, x_test, y_test)

    threads.append(threading.Thread(target=determine_parameters, args=(svm_opt,)))
    threads.append(threading.Thread(target=determine_parameters, args=(ann_opt,)))
    threads.append(threading.Thread(target=determine_parameters, args=(tree_opt,)))
    threads.append(threading.Thread(target=determine_parameters, args=(forest_opt,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    config.svm = svm_opt.svm
    config.ann = ann_opt.ann
    config.decision_tree = tree_opt.decision_tree
    config.random_forest = forest_opt.random_forest

    print config.toDict()

    return config


def determine_parameters(optimizer):
    print 'determine parameters ', optimizer.__class__.__name__
    optimizer.optimize()


def score_model(x_all, y_all, x_holdout, y_holdout, x_without_holdout, y_without_holdout, x_train, y_train, x_test,
                y_test, classifier):
    classifierCV = deepcopy(classifier)

    classifier.fit(x_train, y_train)

    score1 = classifier.score(x_test, y_test)  # score on test data set
    scoresCV = cross_val_score(classifierCV, x_holdout, y_holdout, cv=10)
    score2 = np.mean(scoresCV)  # estimate accuracy using cv with 5000 samples
    score3 = classifier.score(x_without_holdout,
                              y_without_holdout)  # accuracy over entire dataset without holdout samples (5000)
    # score4 = classifier.score(x_train, y_train) # accuracy over train samples (should be the highest)
    # score5 = classifier.score(x_all, y_all) # accuract over entire dataset

    return abs(score3 - score2), abs(score3 - score1)


if __name__ == '__main__':
    main()
