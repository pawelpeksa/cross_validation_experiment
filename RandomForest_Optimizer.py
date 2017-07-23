from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from Optimizer import Optimizer
from MethodsConfiguration import RandomForest

DEPTH_KEY = 'depth'
ESTIMATORS_KEY = 'estimators'


class RandomForest_Optimizer(Optimizer):
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=10,
                 depth_begin=1, depth_end=10,
                 estimators_begin=2, estimators_end=10):
        Optimizer.__init__(self, x_train, y_train, x_test, y_test, n_folds)

        self._depth_begin = depth_begin
        self._depth_end = depth_end
        self._estimators_begin = estimators_begin
        self._estimators_end = estimators_end

        self.random_forest = RandomForest()

        self._init_hyper_space()

    def _init_hyper_space(self):
        self._hyper_space = [hp.choice(DEPTH_KEY, np.arange(self._depth_begin, self._depth_end + 1)),
                             hp.choice(ESTIMATORS_KEY, np.arange(self._depth_begin, self._depth_end + 1))]

    def _objective(self, args):
        Optimizer._print_progress(self, 'random forest')
        depth, estimators = args

        forest = RandomForestClassifier(max_depth=depth, n_estimators=estimators)
        score = Optimizer._objective(self, forest)

        return score

    def optimize(self):
        result = Optimizer.optimize(self)

        self.random_forest.max_depth = result[DEPTH_KEY]
        self.random_forest.n_estimators = result[ESTIMATORS_KEY]
