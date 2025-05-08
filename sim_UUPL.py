from typing import List, Tuple, Dict, Optional
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from util import *
import seaborn as sns
from tqdm import tqdm
import sys
import time
from GP_ours import GaussianProcess


# Define GMM parameters as global constants
MEANS = [np.array([-2, 3]), np.array([0, -3]), np.array([2, 2])]
COVARIANCES = [
    np.array([[2, 1], [1, 2]]),
    np.array([[10, -3], [-3, 4]]),
    np.array([[2, 0], [0, 2]])
]
WEIGHTS = [5/1.6, 22/1.6, 10/1.6]


def findBestQuery(gp: GaussianProcess) -> Tuple[np.ndarray, float]:
    """Find the best query points using optimization.
    
    Args:
        gp: Gaussian Process model
        
    Returns:
        Tuple containing optimal query points and information gain
    """
    def negative_info_gain(x: np.ndarray) -> float:
        return -1 * gp.objectiveEntropy(x)
        
    x0 = np.array(list(gp.initialPoint) * 2) + np.random.uniform(-6, 4, gp.dim * 2)
    opt_res = opt.fmin_l_bfgs_b(
        negative_info_gain, 
        x0=x0, 
        bounds=[(-5, 5)] * gp.dim * 2, 
        approx_grad=True, 
        factr=0.1, 
        iprint=-1
    )
    return opt_res[0], -opt_res[1]


def get_metrics(pos: np.ndarray, gmm_pdf: np.ndarray, gp: GaussianProcess) -> None:
    """Compute and print correlation metrics.
    
    Args:
        pos: Position array
        gmm_pdf: GMM probability density values
        gp: Gaussian Process model
    """
    y_pred = gp.mean1pt(pos.reshape(-1, 2), eval=True)
    corr = np.corrcoef(gmm_pdf.flatten(), y_pred)[0, 1]
    print(">>> corr:", corr)
    corr_list.append(corr)


def create_gmm_surface() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create and visualize the GMM surface.
    
    Returns:
        Tuple containing x, y coordinates and GMM PDF values
    """
    step_size = 0.1
    x, y = np.mgrid[-5:5+step_size:step_size, -5:5+step_size:step_size]
    pos = np.dstack((x, y))
    
    gmm_pdf = np.zeros(x.shape)
    for mean, cov, weight in zip(MEANS, COVARIANCES, WEIGHTS):
        rv = multivariate_normal(mean, cov)
        gmm_pdf += weight * rv.pdf(pos)
    
    # Visualize the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, gmm_pdf, cmap='viridis')
    ax.view_init(elev=90, azim=-90)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Reward')
    plt.show()
    
    return x, y, gmm_pdf


def main():
    """Main function to run the simulation."""
    # Set random seed for reproducibility
    np.random.seed(47)
    
    # Initialize correlation list
    global corr_list
    corr_list = []
    
    # Create GMM surface
    x, y, gmm_pdf = create_gmm_surface()
    
    # Initialize Gaussian Process
    initialPoint = [1, 1]
    theta = 0.5
    noise_level = 0.1
    
    gp = GaussianProcess(initialPoint, theta, noise_level)
    pref_dict = {(2, 2): 1, (0, 0): 1}
    gp.updateParameters([[0, 0], [2, 2]], -1, 5, pref_dict)
    
    # Define uncertainty parameters
    uncertainty_dict = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
    uncertainty_thresh = [0.01, 0.05, 0.1, 0.2]
    
    # Main simulation loop
    for i in tqdm(range(50)):
        start = time.time()
        
        # Find optimal query
        optimal_query, info_gain = findBestQuery(gp)
        next_query_1 = [float(round(optimal_query[0], 1)), float(round(optimal_query[1], 1))]
        next_query_2 = [float(round(optimal_query[2], 1)), float(round(optimal_query[3], 1))]
        
        # Convert to tuples for dictionary keys
        a = (next_query_1[0], next_query_1[1])
        b = (next_query_2[0], next_query_2[1])
        
        # Calculate values for both points
        point = np.array([next_query_1, next_query_2])
        value_q1 = value_q2 = 0
        
        for mean, cov, weight in zip(MEANS, COVARIANCES, WEIGHTS):
            rv = multivariate_normal(mean, cov)
            values = weight * rv.pdf(point)
            value_q1 += values[0]
            value_q2 += values[1]
        
        # Determine uncertainty level
        diff = np.linalg.norm(value_q1 - value_q2)
        uncertainty_level = 1
        for j, thresh in enumerate(uncertainty_thresh, 1):
            if diff > thresh:
                uncertainty_level = j + 1
        
        # Update preference dictionary
        pref_dict[a] = pref_dict.get(a, 0) + uncertainty_dict[uncertainty_level]
        pref_dict[b] = pref_dict.get(b, 0) + uncertainty_dict[uncertainty_level]
        
        # Update GP parameters
        gp.updateParameters(
            [next_query_1, next_query_2],
            1 if value_q1 > value_q2 else -1,
            uncertainty_level,
            pref_dict
        )
        
        # Additional update for low uncertainty
        if uncertainty_level == 1:
            gp.updateParameters(
                [next_query_1, next_query_2],
                1 if value_q1 <= value_q2 else -1,
                uncertainty_level,
                pref_dict
            )
    
    # Final visualization
    pos = np.dstack((x, y))
    Xpred = pos.reshape(-1, 2)  # Reshape for prediction
    y_pred = gp.mean1pt(Xpred, eval=True)  # Get predictions for all points
    y_pred = y_pred.reshape(x.shape)  # Reshape back to match the grid
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, y_pred, cmap='viridis')
    ax.view_init(elev=90, azim=-90)  # Match the view angle of the initial plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('PDF value')
    ax.set_title('Final plot')
    plt.show()


if __name__ == "__main__":
    main()
