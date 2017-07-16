from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

from Optimizer import Optimizer
from Configuration import Configuration

C_KEY = 'C'


class SVM_Optimizer(Optimizer):
    def __init__(self, x, y, n_folds=10, C_begin=2 ** -5, C_end=2):
        Optimizer.__init__(self, x, y, n_folds)

        self._C_begin = C_begin
        self._C_end = C_end

        self._init_hyper_space()
        self._i = 0

    def _init_hyper_space(self):
        self._hyper_space = hp.uniform(C_KEY, self._C_begin, self._C_end)

    def _objective(self, args):
        self._i += 1
        print 'Svm optimizer progress:', str((self._i/float(Configuration.HYPEROPT_EVALS_PER_SEARCH)) * 100), '%'
        C = args
        # train on x_test, y_test, test on x_test, y_test
        # TODO

        SVM = svm.SVC(kernel='linear', C=C)
        # minus because it's minimization and we want to maximize
        score = - (np.mean(cross_val_score(SVM, self._x, self._y, cv=self._n_folds)))

        return score

    def optimize(self):
        result = Optimizer.optimize(self)
        return result[C_KEY]
