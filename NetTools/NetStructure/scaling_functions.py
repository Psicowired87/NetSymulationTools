
"""
Module which contains transformations of the network using global information
but not involving changes in their structures.

"""


import numpy as np


###############################################################################
############################### Transformation ################################
###############################################################################
def scaling_scores(comparison, method, args):
    """Scaling of the scores to other values keeping order. Normally used to
    transform real values values to [0,1]-values.

    Parameters
    ----------
    comparison: array_like, shape (Nelements, Nelements)
        the matrix which represents the network connections.
    method: str, function or list
        the method or methods used to transform scores in order to scale them
        keeping their order.
    args: list of dicts or dict
        the parameters of the methods selected.

    Returns
    -------
    scores: array_like, shape (Nelements, Nelements)
        matrix of connections of the network.

    """

    ## 0. Preparing inputs
    if type(method) != list:
        methods = [method]
    else:
        methods = method
    if type(args) == dict:
        args = [args]

    ## 1. Looping methods
    i = 0
    for method in methods:
        if type(method).__name__ == 'function':
            comparison = method(comparison, **args[i])
        elif method == 'gap_normalization':
            comparison = normalization_scores_min_max(comparison, **args[i])
        elif method == 'thresholding':
            comparison = thresholding_scores(comparison, **args[i])
        elif method == 'standarization':
            comparison = standarization_scores(comparison, **args[i])
        elif method == 'scale_connectome':
            comparison = scale_connectome_like(comparison, **args[i])
        i += 1

    scores = comparison
    return scores


def thresholding_scores(comparison, thr):
    """Transformation of the scores to {0-1}-values by thresholding the values.

    Parameters
    ----------
    comparison: array_like, shape (Nelements, Nelements)
        the matrix which represents the network connections.
    thr: float
        the value of the threshold.

    Returns
    -------
    scores: array_like, shape (Nelements, Nelements)
        matrix of connections of the network.

    """

    scores = comparison <= thr
    return scores


def normalization_scores_min_max(comparison, limits=()):
    """Transformation of the scores to {0-1}-values by thresholding the values.

    Parameters
    ----------
    comparison: array_like, shape (Nelements, Nelements)
        the matrix which represents the network connections.
    limits: tuple, list or array_like
        the values of the down and up limits.

    Returns
    -------
    scores: array_like, shape (Nelements, Nelements)
        matrix of connections of the network.

    """

    # Setting limits
    if not limits:
        limits = (np.min(comparison), np.max(comparison))

    scores = (comparison-limits[0])/(limits[1]-limits[0])
    return scores


def standarization_scores(comparison):
    """Transformation of the scores values related with the standart deviation
    of the whole values.

    Parameters
    ----------
    comparison: array_like, shape (Nelements, Nelements)
        the matrix which represents the network connections.
    limits: tuple, list or array_like
        the values of the down and up limits.

    Returns
    -------
    scores: array_like, shape (Nelements, Nelements)
        matrix of connections of the network.

    """

    m = np.mean(comparison)
    st = np.std(comparison)
    scores = (comparison-m)/st
    return scores


def min_diagonal(X):
    """Fill the diagonal with the minimum value of the matrix. It is used when
    we know that the self interaction has to be minimum or null.
    """
    np.fill_diagonal(X, X.min())
    return X


def scale_connectome_like(X):
    """Scaling function which uses the assumption of no self-connections and
    binary connections.
    """
    scores = normalization_scores_min_max(min_diagonal(X))
    return scores
