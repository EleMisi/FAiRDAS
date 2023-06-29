from typing import Dict, AnyStr, Union

import numpy as np


class Resource(object):
    """
    Stores an ontology resource with its description and scores
    """

    def __init__(self, name: AnyStr, identifier: int, description: AnyStr,
                 scores: Union[Dict[str, int], None] = None, at_bottom_flag: bool = False):
        """
        Instantiates a Label object.

        Args:
            name (str): the name of the label
            identifier (int): an integer that uniquely identifies a ontology label
            description (str): a long textual description (a few sentences) of the label.
            scores (Dict[str,int], Optional): a dict of scores, one for each label, represented with its id
            is_reliable (bool): whether the label is reliable (i.e. can be used for labeling and matchmaking) or not
        """

        self.identifier = identifier
        self.name = name
        self.description = description
        self.scores = scores
        self.at_bottom_flag = at_bottom_flag

    def __repr__(self):
        return f'resource(id: {self.identifier}, name: {self.name}, flag: {self.at_bottom_flag})'


class MatchmakingScore:

    def __call__(self, x: np.array, y: np.array, return_explanations=False):
        """
        Compute the relavance between two score matrices

        Parameters
        ----------
        x: numpy array or equivalent
            First score matrix (n x k, where k is the number of scores)
        y: numpy array or equivalent
            Second score matrix (m x k, where k is the number of scores)

        Returns
        -------
        numpy array:
            the score vector
        list:
            explanations
        """
        raise NotImplementedError('This methods needs to be implemented')


class MatchmakingAlgorithm:

    def __init_(self):
        self.labels = []
        self.relations = []
        self.resources = []

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

    def configure(self):
        """
        Perform any internal configuration here
        """
        raise NotImplementedError('This methods needs to be implemented')

    def matchOne(self, query_scores, query_metadata=None,
                 return_explanations=False):
        """
        Identify relevant resources w.r.t. a single query

        Parameters:
        - query_scores: a dictionary with
          - keys: label idx
          - values: score for each label
        - metadata: any other symbolic information
          - TODO: define format

        Returns:
        - list of resource identifiers
        - explanations (in some format)
        """
        raise NotImplementedError('This methods needs to be implemented')

    def matchMany(self, query_scores, query_metadata, resource_capacity):
        """
        Parameters:
        - query_scores: a dictionary with
          - keys: label idx
          - values: score for each label
        - metadata: any other symbolic information
          - TODO: define format
        - resource_capacity: maximum number of resources to return

        Returns:
        - list of list
          - Each list contains resource idenfiers for a query
        - explanations (in some format)
        """
        raise NotImplementedError('This methods needs to be implemented')
