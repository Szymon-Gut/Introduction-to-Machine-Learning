import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def mixed_metrix(x1, x2, **kwargs):
    """
    Mixed metric that calculates distance between two observations where only some of the attributes have an order.
    The metric is calculated as an l1 norm (manhattan distance) where the distance along each axis is calculated in the following way:

    - If the attribute has an order the distance calculated is the absolute value of the difference between the values
    - If the attribute does not have an order the distance is 0 if the values are the same and 1 otherwise

    It is advisble to preprocess the ordered variables using MinMaxScaler for better results.

    Parameters
    ----------

    x1 (array like):
    First observation.

    x2 (array like):
    Second observation.

    ordered_attr (set):
    Set of indices of attributes that do have an order.

    weights (array like):
    Array of weights that should be applied along each axis.

    Returns
    -------
    Distance between two vectors in mixed metric.
    
    """
    
    # ordered_attr = kwargs['ordered'] if 'ordered' in kwargs else {}
    # ordered_attr = kwargs['ordered']
    # weights = kwargs['weights'] if 'weights' in kwargs else [1] * len(x1)
    distance = 0

    vfunc = np.vectorize(lambda x : 0 if x == 0 else 1)

    # ordered_idx = np.intersect1d(np.arange(len(x1)), np.array(list(ordered_attr)))
    # unordered_idx = np.setdiff1d(np.arange(len(x1)), ordered_idx)

    # distance += np.abs((x2[ordered_idx] - x1[ordered_idx])).sum()
    # distance += vfunc(x2[unordered_idx] - x1[unordered_idx]).sum()
    return vfunc(x2 - x1).sum()

    return distance

def calculate_distance_matrix(observations, ordered = {}):
    m = observations.shape[0]    
    distances = np.empty(shape=(m, m))
    for i in range(m):
        for j in range(m):
            distances[i, j] = mixed_metrix(observations[i, :], observations[j, :], ordered=ordered)
    return distances


def main():
    pass



if __name__ == '__main__':
    print(main())
    # print('a')