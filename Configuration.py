class Configuration:
    def __init__(self):
        pass

    HYPEROPT_EVALS_PER_SEARCH = 200
    ANN_MAX_ITERATIONS = 1000
    ANN_OPIMIZER_MAX_ITERATIONS = 500

    N_SAMPLES = 100000
    RUNS_FOR_SAMPLE = 10
    n_samples_arr = [500, 1000, 1500]
