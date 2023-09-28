from collections import OrderedDict

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from utils import base


class ScalProdScore:
    def __call__(self, x: np.array, y: np.array, return_explanations: bool = False):
        if not return_explanations:
            # Compute scores
            scores = x.dot(y.T)
            return scores
        else:
            # Compute individual terms of the dot product
            pre_dot = x[:, None, :] * y[None, :, :]
            # Compute the scores
            scores = pre_dot.sum(axis=2)
            return scores, pre_dot


class CosineSimilarityScore:
    def __call__(self, x: np.array, y: np.array, return_explanations: bool = False):
        if not return_explanations:
            # Compute dot product
            scores_raw = x.dot(y.T)
            # Compute norms
            normx = np.linalg.norm(x, ord=2, axis=1).reshape(1, -1)
            normy = np.linalg.norm(y, ord=2, axis=1).reshape(1, -1)
            # Compute norm product
            normp = normx.T.dot(normy)
            # Compute actual scores
            scores = scores_raw / normp
            # Return normalized scores
            return scores
        else:
            # Compute individual alignment scores
            pre_dot_raw = x[:, None, :] * y[None, :, :]
            # Compute norms
            normx = np.linalg.norm(x, ord=2, axis=1).reshape(1, -1)
            normy = np.linalg.norm(y, ord=2, axis=1).reshape(1, -1)
            # Compute norm product
            normp = normx.T.dot(normy)
            # Normalize the indidivual scores
            pre_dot = pre_dot_raw / normp[:, :, None]
            # Compute the indivual terms of the dot product
            scores = pre_dot.sum(axis=2)
            return scores, pre_dot


class OnlineMatchmaking(base.MatchmakingAlgorithm):

    def __init__(self, metrics):
        """
        Parameters:
        - metrics: matchmaking distance function
        """
        self.metrics = metrics
        self.labels = []
        self.relations = []
        self.resources = []
        self.vectorizer = None
        self.resource_matrix = None

    def loadLabels(self, labels: list, relations: dict):
        """
        Parameters:
        - labelsAI: a list of Label objects
        - relations: a dictionary with:
          - keys: type of relation
          - values: list of tuples (label1, label2)
        """
        self.labels = labels
        self.relations = relations

    def loadResources(self, resources: list):
        """
        Parameters:
        - resources: a list of resources
        """
        self.resources = resources
        self.resources_scores = [r.scores for r in resources]
        # Convert resource scores to a sparse matrix
        self.vectorizer = DictVectorizer(sparse=True)
        resource_matrix = self.vectorizer.fit_transform(self.resources_scores)
        self.resource_matrix = resource_matrix.toarray()

    def matchOne(self, query_scores: dict, args={}):
        """
        Identify relevant resources w.r.t. a single query

        Parameters:
        - query_scores: a dictionary with
          - keys: label idx
          - values: score for each label
        - args: arguments of the distance function

        Returns:
        - list of ascending ordered (resource_identifier, matching_score)
        """
        # Check resources
        assert len(self.resources) > 0, "No resources found."

        # Transform resource and query into an array
        query_v = self.vectorizer.transform(query_scores)
        query_v = query_v.toarray()

        # Compute matching scores
        res = self.metrics(query_v, self.resource_matrix,
                           return_explanations=False, **args)
        matching_scores = res

        # There's only one query, so let's take the first entry from each result
        matching_scores = matching_scores[0]

        # Sort resource indexes based on the scores
        resource_idxs = np.argsort(-matching_scores)

        # Return list of (resource_identifier, matching_score)
        ordered_matching = [(idx, matching_scores[idx])
                            for idx in resource_idxs]

        return OrderedDict(ordered_matching)
