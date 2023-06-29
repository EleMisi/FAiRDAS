import os
import random

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# Set seed function
def set_seed(seed=424242):
    random.seed(seed)
    np.random.seed(seed)
    # print(f'Random seed {seed} has been set.')
    return seed
