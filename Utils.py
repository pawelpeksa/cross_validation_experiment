import random
import numpy as np


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def remove_random_sample(x, y, sample_n):
        sample = random.sample(range(1, len(x)), sample_n)

        x_u = [x[i] for i in sample]
        y_u = [y[i] for i in sample]

        # x = np.delete(x, sample, 0)
        # y = np.delete(y, sample, 0)

        np.delete(x, sample, 0)
        np.delete(y, sample, 0)

        return x, y, np.array(x_u), np.array(y_u)