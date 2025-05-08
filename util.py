import numpy as np
from scipy.stats import norm


def h(x):
    """
    Binary entropy function: H(x) = -x*log2(x) - (1-x)*log2(1-x)
    
    Args:
        x: Probability value between 0 and 1
        
    Returns:
        float: Binary entropy value
    """
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def phi(x, sigma=1):
    """
    Cumulative distribution function of normal distribution.
    
    Args:
        x: Input value
        sigma: Standard deviation (default=1)
        
    Returns:
        float: CDF value
    """
    return norm.cdf(x, scale=sigma)


def phip(x, sigma=1):
    """
    Probability density function of normal distribution.
    
    Args:
        x: Input value
        sigma: Standard deviation (default=1)
        
    Returns:
        float: PDF value
    """
    return norm.pdf(x, scale=sigma)


def phipp(x, sigma=1):
    """
    Second derivative of normal PDF.
    
    Args:
        x: Input value
        sigma: Standard deviation (default=1)
        
    Returns:
        float: Second derivative value
    """
    return -x/(sigma**2) * np.exp(-x ** 2 / (2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def normpdf(x):
    """
    Normalized probability density function.
    
    Args:
        x: Input value
        
    Returns:
        float: Normalized PDF value
    """
    return np.exp(-x ** 2 * np.pi)
    # return np.exp(-x ** 2 * np.pi / 4) / 2


def normpdf_derivative(x):
    """
    Derivative of normalized probability density function.
    
    Args:
        x: Input value
        
    Returns:
        float: Derivative value
    """
    return -x * np.exp(-x ** 2 * np.pi) * (2 * np.pi)
    # return -np.pi/4 * x * np.exp(-x ** 2 * np.pi / 4)
