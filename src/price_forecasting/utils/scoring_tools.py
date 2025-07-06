from typing import Sequence

import numpy as np
from properscoring import crps_ensemble


def get_mean_crps(
    quantile_preds: Sequence,
    y_true: Sequence,
    quantile_levels: Sequence,
    n_interpolated_samples: int = 1000,
) -> float:
    """Calculate mean CRPS on quantiles by sampling piecewise-linear interpolation.

    Args:
        quantile_preds: 2-D array of quantile predictions
        y_true: Array of target values
        quantile_levels: Array of quantile levels
        n_interpolated_samples: Number of synthetic samples to generate per point.

    Returns:
        float: Mean CRPS score for predictions.

    """
    quantile_levels = np.array(quantile_levels)

    n_points = quantile_preds.shape[0]
    interpolated_samples = np.empty((n_points, n_interpolated_samples))

    for i in range(n_points):
        probs = np.random.uniform(0, 1, n_interpolated_samples)
        samples = np.interp(probs, quantile_levels, quantile_preds[i])
        interpolated_samples[i] = samples

    crps_scores = crps_ensemble(y_true, interpolated_samples)
    return np.mean(crps_scores)
