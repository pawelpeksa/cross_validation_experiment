from hyperopt import fmin, tpe

from Configuration import Configuration


class Optimizer():
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=10):
        self._x_train = x_train
        self._y_train = y_train

        self._x_test = x_test
        self._y_test = y_test

        self._n_folds = n_folds

    def optimize(self):
        return fmin(fn=self._objective, space=self._hyper_space, algo=tpe.suggest,
                    max_evals=Configuration.HYPEROPT_EVALS_PER_SEARCH)

    def _objective(self, classifier):
        classifier.fit(self._x_train, self._y_train)
        return classifier.score(self._x_test, self._y_test)

    def _init_hyper_space(self):
        raise NotImplementedError('Should have implemented this')

    def _objective(self, args):
        raise NotImplementedError('Should have implemented this')
