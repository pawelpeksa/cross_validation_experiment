from hyperopt import space_eval, hp
from sklearn import svm
import numpy as np

from Optimizer import Optimizer
from MethodsConfiguration import SVM

C_KEY = 'C'


class SVM_Optimizer(Optimizer):
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=10, C_begin=2**-5, C_end=2):
        Optimizer.__init__(self, x_train, y_train, x_test, y_test, n_folds)

        self._C_begin = C_begin
        self._C_end = C_end

        self.svm = SVM()

        self._init_hyper_space()


    def _init_hyper_space(self):
        self._hyper_space = hp.uniform(C_KEY, self._C_begin, self._C_end)

    def _objective(self, args):
        Optimizer._print_progress(self, 'svm')
        C = args

        assert C > 0, 'C <= 0'

        SVM = svm.SVC(kernel='linear', C=C)
        score = Optimizer._objective(self, SVM)

        return score

    def optimize(self):
        result = Optimizer.optimize(self)

        self.svm.C = result
