"""Simulation 2 (Tabletop Importance) — Baseline 1: Chu & Ghahramani (2005).

Requires:
    pip install GPro

Reference:
    Wei Chu and Zoubin Ghahramani. "Preference learning with Gaussian processes."
    ICML 2005.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm

from GPro.preference import ProbitPreferenceGP


# Ground-truth GMM constants
MEANS = [np.array([-2, 3]), np.array([0, -3]), np.array([2, 2])]
COVARIANCES = [
    np.array([[2, 1], [1, 2]]),
    np.array([[10, -3], [-3, 4]]),
    np.array([[2, 0], [0, 2]]),
]
WEIGHTS = [5 / 1.6, 22 / 1.6, 10 / 1.6]


def build_gmm_grid(step_size: float = 0.1):
    """Build evaluation grid and compute ground-truth GMM density.

    Args:
        step_size: Grid resolution.

    Returns:
        Tuple of (x, y, pos, gmm_pdf).
    """
    x, y = np.mgrid[-5:5 + step_size:step_size, -5:5 + step_size:step_size]
    pos = np.dstack((x, y))
    gmm_pdf = np.zeros(x.shape)
    for mean, cov, weight in zip(MEANS, COVARIANCES, WEIGHTS):
        rv = multivariate_normal(mean, cov)
        gmm_pdf += weight * rv.pdf(pos)
    return x, y, pos, gmm_pdf


def gmm_value(points: np.ndarray) -> np.ndarray:
    """Evaluate the ground-truth GMM at an array of points.

    Args:
        points: Array of shape (N, 2).

    Returns:
        Array of GMM density values, shape (N,).
    """
    values = np.zeros(len(points))
    for mean, cov, weight in zip(MEANS, COVARIANCES, WEIGHTS):
        rv = multivariate_normal(mean, cov)
        values += weight * rv.pdf(points)
    return values


def build_preference_dataset(X: np.ndarray) -> np.ndarray:
    """Generate preference pairs from the GMM ground truth.

    For each consecutive pair (X[2i], X[2i+1]) the index of the preferred
    point is stored first, following the GPro convention.

    Args:
        X: Feature matrix of shape (2*N, 2).

    Returns:
        M: Preference index matrix of shape (N, 2).
    """
    n_pairs = len(X) // 2
    M = np.zeros((n_pairs, 2), dtype=int)
    for i in range(n_pairs):
        pair = np.array([X[2 * i], X[2 * i + 1]])
        vals = gmm_value(pair)
        if vals[0] >= vals[1]:
            M[i] = [2 * i, 2 * i + 1]
        else:
            M[i] = [2 * i + 1, 2 * i]
    return M


def get_metrics(gpr: ProbitPreferenceGP, pos: np.ndarray,
                gmm_pdf: np.ndarray, corr_list: list) -> float:
    """Compute and print correlation between true GMM and GP predictions.

    Args:
        gpr: Fitted ProbitPreferenceGP model.
        pos: Position grid of shape (H, W, 2).
        gmm_pdf: True GMM density of shape (H, W).
        corr_list: List to append the computed correlation to.

    Returns:
        float: Pearson correlation coefficient.
    """
    y_pred = np.array([gpr.predict(row, return_y_std=False) for row in pos])
    corr = np.corrcoef(gmm_pdf.flatten(), y_pred.flatten())[0, 1]
    print(">>> corr:", corr)
    corr_list.append(corr)
    return corr


def main():
    np.random.seed(47)

    x, y, pos, gmm_pdf = build_gmm_grid(step_size=0.1)

    # Sample 100 random feature points → 50 preference pairs
    X = np.random.uniform(low=-5, high=5, size=(100, 2))
    M = build_preference_dataset(X)

    gpr = ProbitPreferenceGP()
    corr_list = []

    # Incremental fitting: add one preference pair per iteration
    for i in tqdm(range(1, 51)):
        gpr.fit(X[:2 * i], M[:i], f_prior=None)
        get_metrics(gpr, pos, gmm_pdf, corr_list)

    print("Correlations:", corr_list)

    # Final visualisation
    Xpred = pos.reshape(-1, 2)
    y_pred = gpr.predict(Xpred)
    y_pred_mean = np.mean(y_pred, axis=1).reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, y_pred_mean, cmap="viridis")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Reward value")
    ax.set_title("Baseline 1 (Chu & Ghahramani) — learned reward function")
    plt.savefig("sim2_baseline1.pdf")
    plt.show()


if __name__ == "__main__":
    main()
