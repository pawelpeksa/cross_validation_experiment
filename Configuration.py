class Configuration:
    def __init__(self):
        pass

    # HYPEROPT_EVALS_PER_SEARCH = 2000
    HYPEROPT_EVALS_PER_SEARCH = 1500
    ANN_MAX_ITERATIONS = 1000
    ANN_OPIMIZER_MAX_ITERATIONS = 500

    N_SAMPLES = 100000
    RUNS_FOR_SAMPLE = 10
    n_samples_arr = [50, 100, 150, 200, 250, 300, 500, 1000]
