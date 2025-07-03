from properscoring import crps_ensemble
import numpy as np

def get_mean_crps(
    quantile_preds, 
    y_true, 
    quantile_levels,
    n_interpolated_samples=1000,
    random_seed=101
):
    """
    Converts predicted quantiles into mean CRPS via synthetic samples in a piecewise-linear interpolation.

    Args:
        quantile_preds (array-like): 2-D array of quantile predictions
        quantile_levels (array-like): Array of quantile levels corresponding to columns of quantile_preds
        n_interpolated_samples (int): Number of synthetic samples to generate per forecasted point.
        random_seed (int): random seed for reproducibility
        
    Returns:
        float: Mean CRPS score for predictions. 
    """
    np.random.seed(random_seed)
    quantile_levels = np.array(quantile_levels)

    n_points = quantile_preds.shape[0]
    interpolated_samples = np.empty((n_points, n_interpolated_samples))

    for i in range(n_points):
        probs = np.random.uniform(0, 1, n_interpolated_samples)
        samples = np.interp(probs, quantile_levels, quantile_preds[i])
        interpolated_samples[i] = samples

    crps_scores = crps_ensemble(y_true, interpolated_samples)
    return np.mean(crps_scores)
