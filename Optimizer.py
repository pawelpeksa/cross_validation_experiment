from hyperopt import fmin, tpe

from Configuration import Configuration


class Optimizer():
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=10):
        self._x_train = x_train
        self._y_train = y_train

        self._x_test = x_test
        self._y_test = y_test

        self._n_folds = n_folds

        self._iteration = 0

    def optimize(self):
        return fmin(fn=self._objective, space=self._hyper_space, algo=tpe.suggest,
                    max_evals=Configuration.HYPEROPT_EVALS_PER_SEARCH)

    def _objective(self, classifier):
        self._iteration += 1
        classifier.fit(self._x_train, self._y_train)
        return -classifier.score(self._x_test, self._y_test)

    def _print_progress(self, classifier_str):
        print classifier_str, 'optimizer progress:', str(
            (self._iteration / float(Configuration.HYPEROPT_EVALS_PER_SEARCH)) * 100), '%'

    def _init_hyper_space(self):
        raise NotImplementedError('Should have implemented this')
