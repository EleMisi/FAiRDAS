from collections import OrderedDict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import const_define as cd


def power_law(rank: tuple) -> np.array:
    """Compute power law factors based on rank
    :param rank: tuple of resource index
    :return: array of power low factors
    """
    factors = np.ones((len(rank),))
    exp = np.zeros((len(rank),))
    # Interval for the exponents
    sc = MinMaxScaler((1, 10))
    for i in range(len(rank)):
        # Label position in the q_rank
        idx = rank.index(i)
        # Exponent
        exp[i] = len(rank) - idx
    # Scale exponents
    scaled_exp = sc.fit_transform(exp.reshape(-1, 1)).reshape(-1, )
    factors *= np.random.power(scaled_exp)
    return factors


def rank_similarity(rank: tuple, old_rank: tuple) -> np.array:
    """
    Compute cosine similarity distance between the two ranks.
    :param rank:
        OrderedDict with resources rank and scores
    :param old_rank:
        OrderedDict with resources rank and scores
    :return:
        cosine similarity distance
    """
    assert len(rank) == len(old_rank), f'Rank 1 and rank 2 have different lenghts: {len(rank)} and {len(old_rank)}.'
    # Preprocessing rank
    rank1 = tuple([i for i in rank])
    rank2 = tuple([i for i in old_rank])
    # Build vector of position from rank
    n_resources = len(rank1)
    x = np.zeros((n_resources,), dtype=int)
    y = np.zeros((n_resources,), dtype=int)
    for i in range(n_resources):
        x[i] = rank1.index(i)
        y[i] = rank2.index(i)
    similarity = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))
    distance = 1 - similarity
    distance = np.around(distance, 5)
    distance = np.clip(distance, 0, 1)
    # print(distance, similarity)
    return distance


class DIDI:
    """
    Code adapted from
    https://github.com/giuluck/GeneralizedDisparateImpact/blob/2f9ea0973e7e82dda8fcd1c1a275a8e2e3adcfa5/moving_targets/metrics/constraints.py#L40
    """

    def __init__(self, seed: int, resources_scores: np.ndarray, protected_labels: list, impact_function):
        """
        :param resources_scores:
            matrix with the label score (column) for each resource (row)
        :param protected_labels:
            list of protected labels id
        :param impact_function:
            function to compute impact of resource in the ranking
        :param seed: seed for reproducibility
        """

        # Fix seed for reproducibility
        self.seed = self.set_seed(seed)
        self.indicator_matrix = self.get_indicator_matrix(resources_scores, protected_labels)

        self.impact_function = impact_function

    def __call__(self, rank: OrderedDict, old_rank: OrderedDict, args_impact_fn: dict = {}):
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param rank:
            OrderedDict with resources rank and scores
        :return:
            The (absolute) value of the DIDI.
        """
        # Compute impact function
        self.seed = self.set_seed(self.seed)
        output = self.compute_impact_function(rank, **args_impact_fn)
        # Check indicator matrix shape
        assert self.indicator_matrix.shape[1] == output.shape[
            0], f"Wrong number of samples, expected {self.indicator_matrix.shape[1]} got {output.shape[0]}"
        # Compute DIDI
        didi = self.compute_DIDI(output=output)

        return didi

    def compute_DIDI(self, output: np.array) -> float:
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param output:
            array with impact function values
        :return:
            The (absolute) value of the DIDI.
        """
        # Check indicator matrix shape
        assert self.indicator_matrix.shape[1] == output.shape[
            0], f"Wrong number of samples, expected {self.indicator_matrix.shape[1]} got {output.shape[0]}"

        # Compute DIDI
        didi = 0.0
        total_average = np.mean(output)
        # Loop over protected groups
        for protected_group in self.indicator_matrix:
            # Select output of sample belonging to protected attribute
            protected_targets = output[protected_group]
            # Compute partial DIDI over the protected attribute
            if len(protected_targets) > 0:
                protected_average = np.mean(protected_targets)
                didi += abs(protected_average - total_average)
        return didi

    def get_indicator_matrix(self, resources_scores: np.ndarray, protected_labels: list) -> np.array:
        """Computes the indicator matrix given the input data and a protected feature.
        :param resources_scores:
            matrix with the label score (column) for each resource (row)
        :param protected_labels:
            list of protected labels id
        :return:
            indicator matrix, i.e., a matrix in which the i-th row represents a boolean vector stating whether or
            not the j-th sample (represented by the j-th column) is part of the i-th protected group.
        """
        n_samples = resources_scores.shape[0]
        n_groups = len(protected_labels)
        matrix = np.zeros((n_samples, n_groups)).astype(int)
        for i in range(n_groups):
            for j in range(n_samples):
                label = protected_labels[i]
                matrix[j, i] = 1 if resources_scores[j, label] == 1. else 0
        return matrix.transpose().astype(bool)

    def compute_impact_function(self, rank: OrderedDict, args: dict = {}):
        """

        :param rank:
            OrderedDict with resources rank and scores
        :param args:
        :return:
        """
        # Preprocessing rank
        rank = tuple([i for i in rank])
        # Call impact function
        return self.impact_function(rank, **args)

    def set_seed(self, seed):
        """
        Fix seed for reproducibility
        :param seed: seed to be fixed
        :return: fixed seed
        """
        seed = cd.set_seed(seed)
        return seed


class Metrics:

    def __init__(self, metrics_dict: OrderedDict, seed: int):
        """
        :param resources_scores:
            matrix with the label score (column) for each resource (row)
        :param protected_labels:
            list of protected labels id
        :param impact_function:
            function to compute impact of resource in the ranking
        :param seed: seed for reproducibility
        """

        # Fix seed for reproducibility
        self.seed = self.set_seed(seed)
        self.n_metrics = len(metrics_dict)
        self.metrics_dict = metrics_dict

    def compute_one(self, metric_name: str, input: dict):
        """
        Compute the required metric given its input
        """
        return self.metrics_dict[metric_name](*input)

    def compute_all(self, metrics_input: dict):
        """
        Compute all the metrics given their input
        """
        res = {}
        for name, metric in self.metrics_dict.items():
            res[name] = metric(*metrics_input[name])
        return res

    def set_seed(self, seed):
        """
        Fix seed for reproducibility
        :param seed: seed to be fixed
        :return: fixed seed
        """
        seed = cd.set_seed(seed)
        return seed

    def __str__(self):
        s = 'Metrics:\n'
        for n, m in self.metrics_dict.items():
            s += f'\t{n} : {m}\n '
        return s
