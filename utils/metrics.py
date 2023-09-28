from collections import OrderedDict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import const_define as cd


def negative_exponential(rank: tuple) -> np.array:
    """The impact function of each rank entries depends on its position
    :param rank: tuple with rank resources
    :return: array of power low factors
    """
    # Position in the rank
    factors = np.zeros(len(rank))
    for i, idx in enumerate(rank):
        factors[idx] = (1 / np.exp(i))*10
    return factors


def rank_similarity(ranks: list, old_ranks: list) -> np.array:
    """
    Compute cosine similarity distance between the two ranks.
    :param ranks:
        list of OrderedDict with resources rank and scores
    :param old_ranks:
        list of OrderedDict with resources rank and scores
    :return:
        cosine similarity distance (mean on the rank vectors provided)
    """
    assert len(ranks) == len(old_ranks), f'Rank 1 and rank 2 have different lengths: {len(ranks)} and {len(old_ranks)}.'
    # Compute cosine distance for each rank provided
    cosine_distance = []
    for rank, old_rank in zip(ranks, old_ranks):
        # Extract vectors of score from rank
        rank1 = []
        rank2 = []
        for idx,score in rank.items():
            rank1.append(idx)
        for idx,score in old_rank.items():
            rank2.append(idx)
        rank1,rank2 = np.array(rank1),np.array(rank2)
        # Compute cosine similarity
        similarity = cosine_similarity(rank1.reshape(1, -1), rank2.reshape(1, -1))
        distance = 1 - similarity
        # Rounding and clipping
        distance = np.around(distance, 5)
        distance = np.clip(distance, 0, 1)
        # Add distance
        cosine_distance.append(distance)
    return np.mean(cosine_distance)


class DIDI:
    """
    Code adapted from
    https://github.com/giuluck/GeneralizedDisparateImpact/blob/2f9ea0973e7e82dda8fcd1c1a275a8e2e3adcfa5/moving_targets/metrics/constraints.py#L40
    """

    def __init__(self, resources_labels: np.ndarray, protected_labels: list, impact_function):
        """
        :param resources_labels:
            matrix with the label score (column) for each resource (row)
        :param protected_labels:
            list of protected labels id
        :param impact_function:
            function to compute impact of resource in the ranking
        """
        self.impact_function = impact_function
        self.protected_labels = protected_labels
        self.indicator_matrix =self.get_indicator_matrix(resources_labels, self.protected_labels)

    def __call__(self, ranks: list, old_ranks: OrderedDict, args_impact_fn: dict = {}):
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param ranks:
            list of OrderedDicts with resources rank and scores
        :return:
            The (absolute) value of the DIDI.
        """
        n_queries = len(ranks)
        output = []
        for rank in ranks:
            # Compute impact function
            output.append(self.compute_impact_function(rank, **args_impact_fn))
        output = np.array(output).ravel()
        # Compute the indicator matrix
        indicator_matrix = np.tile(self.indicator_matrix,(1,n_queries))
        # Check indicator matrix shape
        assert indicator_matrix.shape[1] == output.shape[
            0], f"Wrong number of samples, expected {indicator_matrix.shape[1]} got {output.shape[0]}"
        # Compute DIDI
        didi = self.compute_DIDI(output=output, indicator_matrix=indicator_matrix)

        return didi

    def compute_DIDI(self, output: np.array, indicator_matrix : np.array) -> float:
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param output:
            array with impact function values (ordered by reources idx)
        :return:
            The (absolute) value of the DIDI.
        """
        # Check indicator matrix shape
        assert indicator_matrix.shape[1] == output.shape[
            0], f"Wrong number of samples, expected {indicator_matrix.shape[1]} got {output.shape[0]}"

        # Compute DIDI
        didi = 0.0
        total_average = np.mean(output)
        # Loop over protected groups
        for protected_group in indicator_matrix:
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
