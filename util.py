import numpy as np
from scipy.stats import norm


def h(x): # binary entropy function
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def phi(x, sigma=1):
    return norm.cdf(x, scale=sigma)

def phip(x, sigma=1):
    return norm.pdf(x, scale=sigma)

def phipp(x, sigma=1):
    return -x/(sigma**2) * np.exp(-x ** 2 / (2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def normpdf(x):
    return np.exp(-x ** 2 * np.pi)
    # return np.exp(-x ** 2 * np.pi / 4) / 2

def normpdf_derivative(x):
    return -x * np.exp(-x ** 2 * np.pi) * (2 * np.pi)
    # return -np.pi/4 * x * np.exp(-x ** 2 * np.pi / 4)
