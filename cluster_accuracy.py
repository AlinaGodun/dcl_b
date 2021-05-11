import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy with the hungarian method.
    code adapted from:
    https://github.com/XifengGuo/IDEC/blob/master/datasets.py
    Parameters:
    -----------------
    y: true labels, numpy.array with shape `(n_samples,)`
    y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    Returns:
    -----------------
    accuracy, in [0,1]
    Raises:
    -----------------
    ValueError: If size of y_true and y_pred do not match
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    y_true = y_true.astype(np.int64)
    if(y_pred.size != y_true.size):
        raise ValueError("y_true and y_pred sizes do not match, they are: {} != {}".format(
            y_pred.size, y_true.size))
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # apply hungarian method
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).transpose()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size