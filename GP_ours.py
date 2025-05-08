from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
from util import *


class GaussianProcess:
    """Gaussian Process implementation for preference learning.
    
    This class implements a Gaussian Process model for learning preferences between pairs of points
    in a multi-dimensional space, with uncertainty handling and preference dictionary support.
    """
    
    def __init__(self, initialPoint: Union[List[float], np.ndarray] = 0, 
                 theta: float = 0.1, 
                 noise_level: float = 0.1) -> None:
        """Initialize the Gaussian Process model.
        
        Args:
            initialPoint: Initial point in the space
            theta: Length scale parameter for the kernel
            noise_level: Noise level parameter
        """
        self.listQueries: List[List] = []
        self.K: np.ndarray = np.zeros((2, 2))
        self.Kinv: np.ndarray = np.zeros((2, 2))
        self.fqmean: float = 0
        self.theta: float = theta
        self.W: np.ndarray = np.zeros((2, 2))
        self.noise: float = noise_level
        self.initialPoint: np.ndarray = np.array(initialPoint)
        self.dim: int = len(self.initialPoint)
        self.pref_dict: Dict[Tuple[float, float], float] = {}
        self.uncertainty_level: int = 0
        self.uncertainty_sigma_dict: Dict[int, float] = {
            1: 0.01, 2: 0.66, 3: 1.7, 4: 3.35, 5: 9.0
        }

    def updateParameters(self, query: List[List[float]], answer: int, 
                        uncertainty: int, pref_dict: Dict[Tuple[float, float], float]) -> None:
        """Update the model parameters with new query information.
        
        Args:
            query: List of two points to compare
            answer: Preference answer (-1 or 1)
            uncertainty: Uncertainty level (1-5)
            pref_dict: Dictionary of preferences
        """
        self.listQueries.append([query[0], query[1], answer, uncertainty])
        self.uncertainty_level = uncertainty
        self.K = self.covK()
        self.Kinv = inv(self.K + np.identity(2 * len(self.listQueries)) * 1e-8)
        self.fqmean = self.meanmode()
        self.W = self.hessian()
        self.pref_dict = pref_dict

    def objectiveEntropy(self, x: np.ndarray) -> float:
        """Compute the objective function (entropy) for a query [xa,xb].
        
        Args:
            x: Concatenated array of two points [xa, xb]
            
        Returns:
            float: Entropy value
        """
        xa = x[:self.dim]
        xb = x[self.dim:]

        matCov = self.postcov(xa, xb)
        mua, mub = self.postmean(xa, xb)
        sigmap = np.sqrt(np.pi * np.log(2) / 2) * self.noise

        result1 = h(phi((mua - mub) / (np.sqrt(2*self.noise**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))))
        result2 = sigmap * 1 / (np.sqrt(sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1])) * np.exp(
            -0.5 * (mua - mub)**2 / (sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))

        return result1 - result2

    def GMM(self, xa: np.ndarray, xb: np.ndarray) -> Tuple[float, float]:
        """Compute Gaussian Mixture Model weights for two points.
        
        Args:
            xa: First point
            xb: Second point
            
        Returns:
            Tuple of weights for xa and xb
        """
        total_pdf_a = 0
        total_pdf_b = 0
        cov = 1/(np.sqrt(2*np.pi))
        for pref, count in self.pref_dict.items():
            rv = multivariate_normal(np.array(pref), cov)
            total_pdf_a += count * rv.pdf(xa)
            total_pdf_b += count * rv.pdf(xb)
        total_pdf_a += 1
        total_pdf_b += 1
        return 1/total_pdf_a, 1/total_pdf_b

    def kernel(self, xa: Union[List[float], np.ndarray], 
               xb: Union[List[float], np.ndarray]) -> float:
        """Compute the kernel function between two points.
        
        Args:
            xa: First point
            xb: Second point
            
        Returns:
            float: Kernel value
        """
        ker = 1*(np.exp(-self.theta*np.linalg.norm(np.array(xa) - np.array(xb)) ** 2))
        try:
            ker = ker[0]
        except:
            pass
        if ker < 0:
            print("You can not have a negative kernel!")
            exit()
        return ker

    def batch_kernel(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        """Compute kernel values for a batch of points.
        
        Args:
            xa: Array of first points
            xb: Second point
            
        Returns:
            np.ndarray: Array of kernel values
        """
        num = xa.shape[0]
        xb = np.repeat(np.array(xb).reshape(1, -1), num, axis=0)
        return np.exp(-self.theta*np.linalg.norm(np.array(xa) - np.array(xb), axis=1) ** 2)

    def meanmode(self) -> np.ndarray:
        """Find the posterior means for the queries using optimization.
        
        Returns:
            np.ndarray: Array of posterior means
        """
        n = len(self.listQueries)
        Kinv = self.Kinv
        listResults = np.array([q[2] for q in self.listQueries])
        sigmas = np.array([self.uncertainty_sigma_dict[q[3]] for q in self.listQueries])

        def logposterior(f: np.ndarray) -> float:
            """Compute the log posterior probability.
            
            Args:
                f: Function values
                
            Returns:
                float: Log posterior probability
            """
            fodd = f[1::2]
            feven = f[::2]
            fint = 1/self.noise*(feven-fodd)
            res = np.multiply(fint, listResults)
            res = res.astype(dtype=np.float64)
            res = norm.cdf(res, scale=sigmas[:n])
            res[res == 0] = 1e-100
            res = np.log(res)
            res = np.sum(res)
            ftransp = f.reshape(-1,1)
            return -1*(res- 0.5 * np.matmul(f, np.matmul(Kinv, ftransp)))

        def gradientlog(f: np.ndarray) -> np.ndarray:
            """Compute the gradient of the log posterior probability.
            
            Args:
                f: Function values
                
            Returns:
                np.ndarray: Gradient vector
            """
            grad = np.zeros(2*len(self.listQueries))
            for i in range(len(self.listQueries)):
                signe = self.listQueries[i][2]
                diff = f[2*i]-f[2*i+1]
                temp = phi(signe*1/self.noise*(diff), sigma=sigmas[i])
                if temp == 0:
                    temp = 1e-100
                grad[2*i]= signe*(phip(signe*1/self.noise*(diff), sigma=sigmas[i])*1/self.noise)/temp
                grad[2*i+1] = signe*(-phip(signe*1/self.noise*(diff), sigma=sigmas[i])*1/self.noise)/temp
            grad = grad - f@Kinv
            return -grad

        x0 = np.zeros(2*n)
        return minimize(logposterior, x0=x0, jac=gradientlog).x

    def hessian(self) -> np.ndarray:
        """Compute the Hessian matrix.
        
        Returns:
            np.ndarray: Hessian matrix
        """
        n = len(self.listQueries)
        W = np.zeros((2*n,2*n))
        for i in range(n):
            dif = self.listQueries[i][2]*1/self.noise*(self.fqmean[2*i]-self.fqmean[2*i+1])
            W[2*i][2*i] = -(1/self.noise**2)*(phipp(dif)*phi(dif)-phip(dif)**2)/(phi(dif)**2)
            W[2*i+1][2*i] = -W[2*i][2*i]
            W[2*i][2*i+1] = -W[2*i][2*i]
            W[2*i+1][2*i+1] = W[2*i][2*i]
        return W

    def kt(self, xa: np.ndarray, xb: np.ndarray, eval: bool = False) -> np.ndarray:
        """Compute covariance between points and queries.
        
        Args:
            xa: First point
            xb: Second point
            eval: Whether in evaluation mode
            
        Returns:
            np.ndarray: Covariance matrix
        """
        n = len(self.listQueries)
        if eval:
            return np.array([self.batch_kernel(xa,self.listQueries[i][j])for i in range(n) for j in range(2)])
        else:
            return np.array([[self.kernel(xa,self.listQueries[i][j])for i in range(n) for j in range(2)], [self.kernel(xb,self.listQueries[i][j])for i in range(n) for j in range(2)]])

    def covK(self) -> np.ndarray:
        """Compute covariance matrix for all queries.
        
        Returns:
            np.ndarray: Covariance matrix
        """
        n = len(self.listQueries)
        return np.array([[self.kernel(self.listQueries[i][j], self.listQueries[l][m]) for l in range(n) for m in range(2)] for i in range(n) for j in range(2)])

    def postmean(self, xa: np.ndarray, xb: np.ndarray, eval: bool = False) -> np.ndarray:
        """Compute posterior mean vector for two points.
        
        Args:
            xa: First point
            xb: Second point
            eval: Whether in evaluation mode
            
        Returns:
            np.ndarray: Posterior mean vector
        """
        kt = self.kt(xa,xb, eval=eval)
        if eval == True:
            kt = kt.T
        return np.matmul(kt, np.matmul(self.Kinv,self.fqmean))

    def cov1pt(self, x: np.ndarray) -> float:
        """Compute variance for a single point.
        
        Args:
            x: Input point
            
        Returns:
            float: Variance value
        """
        return self.postcov(x,0)[0][0]

    def mean1pt(self, x: np.ndarray, eval: bool = False) -> Union[float, np.ndarray]:
        """Compute mean for a single point.
        
        Args:
            x: Input point
            eval: Whether in evaluation mode
            
        Returns:
            Union[float, np.ndarray]: Mean value(s)
        """
        if eval:
            return self.postmean(x,0, eval=eval)
        else:
            return self.postmean(x,0, eval=eval)[0]

    def postcov(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        """Compute posterior covariance matrix for two points.
        
        Args:
            xa: First point
            xb: Second point
            
        Returns:
            np.ndarray: Posterior covariance matrix
        """
        n = len(self.listQueries)
        Kt = np.array([[self.kernel(xa,xa), self.kernel(xa,xb)], [self.kernel(xb,xa), self.kernel(xb,xb)]])
        kt = self.kt(xa,xb)
        W = self.W
        K = self.K
        post_cov = Kt - kt@inv(np.identity(2*n)+np.matmul(W,K))@W@np.transpose(kt)
        xaa, xbb = self.GMM(xa, xb)
        post_cov[0][0] *= xaa
        post_cov[0][0] *= xaa
        post_cov[0][1] *= xaa
        post_cov[0][1] *= xbb
        post_cov[1][0] *= xaa
        post_cov[1][0] *= xbb
        post_cov[1][1] *= xbb
        post_cov[1][1] *= xbb
        return post_cov
