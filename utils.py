import random
import numpy as np
import pandas as pd


def is_empty(x):
    if isinstance(x, (list, dict, str)):
        return len(x) == 0
    return pd.isna(x)

def set_random_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    return seed

def get_random_seed(base_seed=None, fixed_range=1000):
    if base_seed:
        set_random_seed(base_seed)
    return random.randint(0, fixed_range)