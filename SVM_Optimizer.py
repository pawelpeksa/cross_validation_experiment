from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

from Optimizer import Optimizer
from Configuration import Configuration

C_KEY = 'C'


class SVM_Optimizer:
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=10, C_begin=2**-5, C_end=2):

        self._C_begin = C_begin
        self._C_end = C_end

        self._init_hyper_space()
        self._i = 0

        self._x_train = x_train
        self._y_train = y_train

        self._x_test = x_test
        self._y_test = y_test

    def _init_hyper_space(self):
        self._hyper_space = hp.uniform(C_KEY, self._C_begin, self._C_end)

    def _objective(self, args):
        self._i += 1
        print 'Svm optimizer progress:', str((self._i/float(Configuration.HYPEROPT_EVALS_PER_SEARCH)) * 100), '%'
        C = args

        SVM = svm.SVC(kernel='linear', C=C)
        SVM.fit(self._x_train, self._y_train)
        # minus because it's minimization and we want to maximize
        score = - (SVM.score(self._x_test, self._y_test) + 0.5 * SVM.score(self._x_train, self._y_train))

        return score

    def optimize(self):
        result = fmin(fn=self._objective, space=self._hyper_space, algo=tpe.suggest, max_evals=Configuration.HYPEROPT_EVALS_PER_SEARCH)
        return result[C_KEY]
