import numpy as np
from typing import Optional, Union, Tuple

def matrix_wrapper(distance):
    def wrapper(x: np.array, y: np.array):
        ind = (x != 0) * (y != 0)
        return distance(x[ind], y[ind])
    return wrapper

@matrix_wrapper
def euclidean_distance(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean distance between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Length of the line segment connecting given points
    """
    return np.sqrt(np.square(x-y).sum())


def euclidean_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean similarity between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Similarity between points x and y
    """
    return 1 / (1 + euclidean_distance(x, y))

@matrix_wrapper
def pearson_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate a Pearson correlation coefficient given 1-D data arrays x and y
    Args:
        x, y: two points in n-space
    Returns:
        Pearson correlation between x and y
    """
    x = x - x.mean()
    y = y - y.mean()
    return ( x * y ).sum() / np.sqrt(np.square(x).sum()) / np.sqrt(np.square(y).sum())


def apk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the average precision at k
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    score, cnt = 0, 0
    predicted = predicted[:k]
    for i in range(len(predicted)):
        if predicted[i] in actual and predicted[i] not in predicted[:i]:
            cnt += 1
            score += cnt / (i + 1)
    return float(score) / min(len(actual), k)


def mapk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the mean average precision at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    return np.array([apk(actual[i], predicted[i], k) for i in range(predicted.shape[0])]).mean()
