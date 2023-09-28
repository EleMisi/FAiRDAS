from collections import OrderedDict
from typing import List

from utils.base import Resource


def at_bottom(ranks: list, resources: List[Resource]):
    """
    Perform "at the bottom" action for a list of resources
    :param ranks:
        list of OrderedDict with resources rank and scores
    :param resources:
        list of Resource objects
    :return:
        list of resources rank and list of flag resources indices
    """
    # List idx of flagged resources
    flagged_resources = [r.identifier for r in resources if r.at_bottom_flag]
    new_ranks = []
    for rank in ranks:
        # Copy the existing rank without flagged resources
        new_rank = OrderedDict([(idx, rank[idx]) for idx in rank if idx not in flagged_resources])
        # Add flagged resources at the bottom of the new rank
        for idx in flagged_resources:
            new_rank[idx] = rank[idx]
        new_ranks.append(new_rank)

    return new_ranks, flagged_resources
