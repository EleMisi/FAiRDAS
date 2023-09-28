from collections import OrderedDict
from typing import List

import numpy as np

from utils import actions
from utils.base import Resource
from utils.metrics import Metrics


def obj_fn_l2(flag, x: np.array, rank: OrderedDict, resources_list: List[Resource], metrics: Metrics,
              metrics_weight: dict, args_metrics: dict = {}):
    """
    Compute the squared error between real synamic state and approximated one.
    :param flag: array of flags for the action
    :param x: dynamical state to approximate
    :param rank: current resources rank
    :param resources_list: list of Resource objects
    :param metrics: fairness metrics object
    :param metrics_weight: dict of weights to aggregate the metrics
    :param args_metrics: additional arguments of the fairness metrics
    :return: approximation error, real state, approximate state
    """
    # Change flag of resources
    resources_list_flagged = [Resource(name=resources_list[i].name, identifier=resources_list[i].identifier,
                                       description=resources_list[i].description,
                                       scores=resources_list[i].scores, at_bottom_flag=flag[i]) for i in
                              range(len(flag))]
    # Compute new rank
    new_rank, actioned_resources = actions.at_bottom(rank, resources_list_flagged)
    x_approx = np.zeros((metrics.n_metrics,))
    weights = np.zeros((metrics.n_metrics,))
    for i, name in enumerate(metrics.metrics_dict):
        x_approx[i] = metrics.metrics_dict[name](new_rank, old_rank=rank, **args_metrics[name])
        # Populate weight vector
        weights[i] = metrics_weight[name]

    # Compute distance
    diff = x_approx - x
    # Square diff
    squared_diff = diff ** 2
    # Weighted sum
    cost = np.sum(squared_diff * weights) ** 0.5
    # cost = np.linalg.norm(diff)
    return cost, x_approx, new_rank


def obj_fn_l2_wHistory(flag, x: np.array,  ranks: List,
                       resources_list: List[Resource], metrics: Metrics, metrics_weight: dict,
                       args_metrics: dict = {}):
    """
    Compute the squared error between real synamic state and approximated one.
    :param flag:
        array of flags for the action
    :param x:
        dynamical state to approximate
    :param ranks:
        list of current batch ranks
    :param resources_list:
        list of Resource objects
    :param metrics:
        fairness metrics object
    :param metrics_weight:
        dict of weights to aggregate the metrics
    :param args_metrics:
        additional arguments of the fairness metrics
    :return:
        approximation error, real state, approximate state
    """

    # Change flag of resources
    resources_list_flagged = [Resource(name=resources_list[i].name, identifier=resources_list[i].identifier,
                                       description=resources_list[i].description,
                                       scores=resources_list[i].scores, at_bottom_flag=flag[i]) for i in
                              range(len(flag))]
    # Compute new rank
    new_ranks, actioned_resources = actions.at_bottom(ranks, resources_list_flagged)

    x_approx = np.zeros((metrics.n_metrics,))
    weights = np.zeros((metrics.n_metrics,))
    for i, name in enumerate(metrics.metrics_dict):
        x_approx[i] = metrics.metrics_dict[name](ranks=new_ranks, old_ranks=ranks, **args_metrics[name])
        # Populate weight vector
        weights[i] = metrics_weight[name]

    # Compute distance
    diff = x_approx - x
    # Square diff
    diff *= diff
    # Weighted sum
    current_cost = np.sum(diff * weights) ** 0.5

    return current_cost, x_approx, new_ranks


def obj_fn_l2_only_DIDI(flag, x: np.array, rank: OrderedDict, resources_list: List[Resource], fairness_metrics,
                        args_metrics: dict = {},
                        args_impact_fn: dict = {}):
    """
    Compute the squared error between real synamic state and approximated one.
    :param flag: array of flags for the action
    :param x: dynamical state to approximate
    :param rank: current resources rank
    :param resources_list: list of Resource objects
    :param fairness_metrics: fairness metrics object
    :param args_metrics: additional arguments of the fairness metrics
    :param args_impact_fn: additional arguments of the impact function
    :return: approximation error, real state, approximate state
    """
    # Change flag of resources
    resources_list_flagged = [Resource(name=resources_list[i].name, identifier=resources_list[i].identifier,
                                       description=resources_list[i].description,
                                       scores=resources_list[i].scores, at_bottom_flag=flag[i]) for i in
                              range(len(flag))]
    # Compute new rank
    new_rank, _ = actions.at_bottom(rank, resources_list_flagged)
    # Compute impact function
    output = fairness_metrics.compute_impact_function(new_rank, **args_impact_fn)
    # Compute new fairness metrics value
    x_approx = fairness_metrics.compute_DIDI(output=output, **args_metrics)

    # Compute distance
    diff = x_approx - x
    cost = np.linalg.norm(diff)

    return cost, x_approx, new_rank
