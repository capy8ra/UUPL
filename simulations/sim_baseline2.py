"""Simulation 2 (Tabletop Importance) — Baseline 2: Benavoli & Azzimonti (2024).

Requires the prefGP library:
    git clone https://github.com/benavoli/prefGP.git
    cd prefGP
    pip install -r requirements.txt

This script must be run from inside the cloned prefGP directory so that the
local package imports (model/, kernel.py, utility/) resolve correctly:

    cd prefGP
    python ../simulations/sim_baseline2.py

Reference:
    Alessio Benavoli and Dario Azzimonti. "A tutorial on learning from
    preferences and choices with Gaussian processes." arXiv:2403.11782, 2024.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm

from model.exactPreference import exactPreference
from kernel import RBF
from utility import paramz


# Ground-truth GMM constants
MEANS = [np.array([-2, 3]), np.array([0, -3]), np.array([2, 2])]
COVARIANCES = [
    np.array([[2, 1], [1, 2]]),
    np.array([[10, -3], [-3, 4]]),
    np.array([[2, 0], [0, 2]]),
]
WEIGHTS = [5 / 1.6, 22 / 1.6, 10 / 1.6]


def build_gmm_grid(step_size: float = 0.5):
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


def build_preference_pairs(X: np.ndarray, n_pairs: int) -> np.ndarray:
    """Sample random preference pairs from the GMM ground truth.

    Each pair stores the index of the preferred point first, following the
    prefGP convention. Relies on the global numpy random state — call
    np.random.seed() before this function to ensure reproducibility.

    Args:
        X: Feature matrix of shape (N, 2).
        n_pairs: Number of preference pairs to generate.

    Returns:
        Pairs: Index matrix of shape (n_pairs, 2).
    """
    pairs = []
    for _ in range(n_pairs):
        i = np.random.randint(X.shape[0])
        j = np.random.randint(X.shape[0])
        vals = gmm_value(np.array([X[i], X[j]]))
        if vals[0] >= vals[1]:
            pairs.append([i, j])
        else:
            pairs.append([j, i])
    return np.array(pairs)


def make_kernel_params(X: np.ndarray) -> dict:
    """Build the kernel parameter dictionary for prefGP.

    Args:
        X: Feature matrix used to infer the number of dimensions.

    Returns:
        dict: Kernel parameter dictionary compatible with prefGP.
    """
    return {
        "lengthscale": {
            "value": 1.5 * np.ones(X.shape[1], float),
            "range": np.vstack([[0.1, 20.0]] * X.shape[1]),
            "transform": paramz.logexp(),
        },
        "variance": {
            "value": np.array([1.0]),
            "range": np.vstack([[1.0, 1.0001]]),
            "transform": paramz.logexp(),
        },
    }


def get_metrics(predictions: np.ndarray, gmm_pdf: np.ndarray,
                corr_list: list) -> float:
    """Compute and print correlation between true GMM and GP predictions.

    Args:
        predictions: Posterior samples from prefGP, shape (N, n_samples).
        gmm_pdf: True GMM density, flattened to match prediction order.
        corr_list: List to append the computed correlation to.

    Returns:
        float: Pearson correlation coefficient.
    """
    y_pred_mean = np.mean(predictions, axis=1)
    corr = np.corrcoef(gmm_pdf.flatten(), y_pred_mean.flatten())[0, 1]
    print(">>> corr:", corr)
    corr_list.append(corr)
    return corr


def main():
    np.random.seed(42)

    # Training grid (coarser, matching original notebook)
    _, _, pos_train, gmm_pdf_train = build_gmm_grid(step_size=0.5)
    X = pos_train.reshape(-1, 2)
    Pairs = build_preference_pairs(X, n_pairs=50)

    corr_list = []
    time_list = []

    # Incremental fitting: add one preference pair per iteration
    for i in tqdm(range(1, 51)):
        start = time.time()

        data = {"Pairs": Pairs[:i], "X": X}
        model = exactPreference(data, RBF, make_kernel_params(X))
        model.sample(nsamples=1000, tune=125)
        predictions = model.predict(X)

        time_list.append(time.time() - start)
        get_metrics(predictions, gmm_pdf_train, corr_list)

    print("Correlations:", corr_list)
    print("Times (s):   ", time_list)

    # Final visualisation on finer grid
    step_size = 0.1
    x, y = np.mgrid[-5:5 + step_size:step_size, -5:5 + step_size:step_size]
    Xpred = np.dstack((x, y)).reshape(-1, 2)
    y_pred = model.predict(Xpred)
    y_pred_mean = np.mean(y_pred, axis=1).reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, y_pred_mean, cmap="viridis")
    ax.view_init(elev=90, azim=-90)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Reward value")
    ax.set_title("Baseline 2 (Benavoli & Azzimonti) — learned reward function")
    plt.savefig("sim2_baseline2.pdf")
    plt.show()


if __name__ == "__main__":
    main()
