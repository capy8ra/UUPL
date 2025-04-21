from GP_baseline3 import GaussianProcess
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from util import *
import seaborn as sns
from tqdm import tqdm
import sys
import time

np.random.seed(47)

def findBestQuery(gp):
    def negative_info_gain(x):
        return -1*gp.objectiveEntropy(x)
    x0 = np.array(list(gp.initialPoint)*2) + np.random.uniform(-6, 4, gp.dim*2)
    opt_res = opt.fmin_l_bfgs_b(negative_info_gain, x0=x0, bounds=[(-5,5)]*gp.dim*2, approx_grad=True, factr=0.1, iprint=-1)
    return opt_res[0], -opt_res[1]


means = [np.array([-2, 3]), np.array([0, -3]), np.array([2, 2])]
covariances = [np.array([[2, 1], [1, 2]]), np.array([[10, -3], [-3, 4]]), np.array([[2, 0], [0, 2]])]
weights = [5/1.6, 22/1.6, 10/1.6]
step_size = 0.1
x, y = np.mgrid[-5:5+step_size:step_size, -5:5+step_size:step_size]
pos = np.dstack((x, y))
# print(pos.shape)
gmm_pdf = np.zeros(x.shape)
for mean, cov, weight in zip(means, covariances, weights):
    rv = multivariate_normal(mean, cov)
    gmm_pdf += weight * rv.pdf(pos)
# print(gmm_pdf)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, gmm_pdf, cmap='viridis')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('PDF value')
# ax.set_title('Reward')
# plt.show()

corr_list = []


def get_metrics(pos, gmm_pdf):
    y_pred = np.array([np.array([gp.mean1pt(yy) for yy in xx]) for xx in pos]).flatten()
    shift = np.mean(gmm_pdf.flatten() - y_pred)
    new_y_pred = y_pred + shift
    corr = np.corrcoef(gmm_pdf.flatten(), new_y_pred)[0, 1]
    print(">>> corr:", corr)
    corr_list.append(corr)


initialPoint = [1, 1] # where we assume the function value is 0
theta = 0.5 # hyperparameter
noise_level = 0.1 # noise parameter that corresponds to \sqrt{2}\sigma in the paper

gp = GaussianProcess(initialPoint, theta, noise_level)
gp.updateParameters([[0, 0], [2, 2]], -1)


for i in tqdm(range(50)):
    start = time.time()
    optimal_query, info_gain = findBestQuery(gp)
    next_query_1 = [float(round(optimal_query[0], 1)), float(round(optimal_query[1], 1))]
    next_query_2 = [float(round(optimal_query[2], 1)), float(round(optimal_query[3], 1))]
    a = (next_query_1[0], next_query_1[1])
    b = (next_query_2[0], next_query_2[1])
    point = np.array([next_query_1, next_query_2])
    value_q1 = 0
    value_q2 = 0
    for mean, cov, weight in zip(means, covariances, weights):
        rv = multivariate_normal(mean, cov)
        values = weight * rv.pdf(point)
        value_q1 += values[0]
        value_q2 += values[1]
    gp.updateParameters([next_query_1, next_query_2], 1 if value_q1 > value_q2 else -1)


# Plotting
y_pred = np.array([np.array([gp.mean1pt(yy) for yy in xx]) for xx in pos])
pos = np.dstack((x, y))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, y_pred, cmap='viridis')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('PDF value')
ax.set_title('Final plot')
plt.show()
# file.close()
