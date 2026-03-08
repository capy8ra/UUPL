from uupl.gp_baseline3 import GaussianProcess
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from uupl.utils import *
from tqdm import tqdm
import time


# Ground-truth GMM constants shared across helpers
MEANS = [np.array([-2, 3]), np.array([0, -3]), np.array([2, 2])]
COVARIANCES = [
    np.array([[2, 1], [1, 2]]),
    np.array([[10, -3], [-3, 4]]),
    np.array([[2, 0], [0, 2]]),
]
WEIGHTS = [5 / 1.6, 22 / 1.6, 10 / 1.6]


def findBestQuery(gp: GaussianProcess):
    """Find the optimal query points that maximize information gain.

    Args:
        gp: GaussianProcess instance

    Returns:
        tuple: (optimal query points as 1-D array, information gain as float)
    """
    def negative_info_gain(x):
        return -1 * gp.objectiveEntropy(x)

    x0 = np.array(list(gp.initialPoint) * 2) + np.random.uniform(-6, 4, gp.dim * 2)
    opt_res = opt.fmin_l_bfgs_b(
        negative_info_gain,
        x0=x0,
        bounds=[(-5, 5)] * gp.dim * 2,
        approx_grad=True,
        factr=0.1,
        iprint=-1,
    )
    return opt_res[0], -opt_res[1]


def get_metrics(pos: np.ndarray, gmm_pdf: np.ndarray,
                gp: GaussianProcess, corr_list: list) -> float:
    """Compute and print correlation between true GMM and GP predictions.

    Args:
        pos: Position grid (shape [H, W, 2])
        gmm_pdf: True GMM density values (shape [H, W])
        gp: Gaussian Process model
        corr_list: List to append the computed correlation to

    Returns:
        float: Pearson correlation coefficient
    """
    y_pred = np.array(
        [np.array([gp.mean1pt(yy) for yy in xx]) for xx in pos]
    ).flatten()
    shift = np.mean(gmm_pdf.flatten() - y_pred)
    new_y_pred = y_pred + shift
    corr = np.corrcoef(gmm_pdf.flatten(), new_y_pred)[0, 1]
    print(">>> corr:", corr)
    corr_list.append(corr)
    return corr


def gmm_value(point: np.ndarray) -> np.ndarray:
    """Evaluate the ground-truth GMM at one or more points.

    Args:
        point: Array of shape (N, dim) or (dim,)

    Returns:
        np.ndarray: GMM density value(s)
    """
    values = np.zeros(point.shape[0] if point.ndim > 1 else 1)
    for mean, cov, weight in zip(MEANS, COVARIANCES, WEIGHTS):
        rv = multivariate_normal(mean, cov)
        values += weight * rv.pdf(point)
    return values


def main():
    np.random.seed(47)

    # Build evaluation grid
    step_size = 0.1
    x, y = np.mgrid[-5:5 + step_size:step_size, -5:5 + step_size:step_size]
    pos = np.dstack((x, y))

    # Ground-truth GMM density on the grid
    gmm_pdf = np.zeros(x.shape)
    for mean, cov, weight in zip(MEANS, COVARIANCES, WEIGHTS):
        rv = multivariate_normal(mean, cov)
        gmm_pdf += weight * rv.pdf(pos)

    corr_list = []

    # Initialise GP with one seed preference
    initialPoint = [1, 1]
    gp = GaussianProcess(initialPoint, theta=0.5, noise_level=0.1)
    gp.updateParameters([[0, 0], [2, 2]], -1)

    # Active preference learning loop
    for i in tqdm(range(50)):
        optimal_query, _ = findBestQuery(gp)
        next_query_1 = [float(round(optimal_query[0], 1)),
                        float(round(optimal_query[1], 1))]
        next_query_2 = [float(round(optimal_query[2], 1)),
                        float(round(optimal_query[3], 1))]

        point = np.array([next_query_1, next_query_2])
        values = gmm_value(point)
        value_q1, value_q2 = values[0], values[1]

        gp.updateParameters(
            [next_query_1, next_query_2],
            1 if value_q1 > value_q2 else -1,
        )

    # Final visualisation
    y_pred = np.array(
        [np.array([gp.mean1pt(yy) for yy in xx]) for xx in pos]
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, y_pred, cmap="viridis")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Reward value")
    ax.set_title("Baseline 3 — learned reward function")
    plt.show()


if __name__ == "__main__":
    main()
