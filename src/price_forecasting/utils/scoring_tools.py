from typing import Sequence

import numpy as np
from properscoring import crps_ensemble


def get_mean_crps(
    y_pred: Sequence,
    y_test: Sequence,
    quantiles: Sequence,
) -> float:
    """Calculate mean CRPS on quantiles by sampling piecewise-linear interpolation.

    Args:
        y_pred: 2-D array of quantile predictions
        y_test: Array of target values
        quantile_levels: Array of quantile levels
        n_interpolated_samples: Number of synthetic samples to generate per point.

    Returns:
        float: Mean CRPS score for predictions.

    """
    scores = []
    for i in range(len(y_test)):
        yp = y_pred[i]
        yt = y_test[i]

        yp = np.sort(yp)
        p = np.interp(yt, yp, quantiles, left=0.0, right=1.0)

        if p == 1.0:
            i = len(yp)
        elif p == 0.0:
            i = 0
        else:
            i = np.where(yp >= yt)[0][0]

        x = np.insert(yp, i, yt)
        x = np.insert(x, i, yt)

        y = np.insert(quantiles, i, p)
        y[i:] = 1 - y[i:]
        y = np.insert(y, i, p)

        scores.append(np.trapezoid(y, x))

    return np.mean(scores)
