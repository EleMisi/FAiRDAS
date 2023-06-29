from collections import OrderedDict
from typing import List

from utils.base import Resource


def at_bottom(rank: OrderedDict, resources: List[Resource]):
    """
    Perform "at the bottom" action for a list of resources
    :param rank:
        OrderedDict with resources rank and scores
    :param resources:
        list of Resource objects
    :return:
        new resources rank and list of flag resources indices
    """
    # List idx of flagged resources
    flagged_resources = [r.identifier for r in resources if r.at_bottom_flag]
    # Copy the existing rank without flagged resources
    new_rank = OrderedDict([(idx, rank[idx]) for idx in rank if idx not in flagged_resources])
    # Add flagged resources at the bottom of teh new rank
    for idx in flagged_resources:
        new_rank[idx] = rank[idx]

    return new_rank, flagged_resources
