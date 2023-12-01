import numpy as np


def get_mean_ui_of_a_time_series(time_series, alpha=0.05):
    """
    :param time_series: (list of np.array)
    :param alpha: (float) significance level
    :return: (tuple) of mean of the timeseries and uncertainty interval of timeseries
    """

    mean = np.mean(time_series, axis=0)
    ui = np.percentile(time_series, q=[alpha / 2 * 100, 100 - alpha / 2 * 100], axis=0)

    return mean, ui
