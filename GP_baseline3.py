from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import norm
from util import *


class GaussianProcess:
    """
    Gaussian Process implementation for preference learning.
    Uses a kernel-based approach to model preferences between pairs of points.
    """
    def __init__(self, initialPoint=0, theta=0.1, noise_level=0.1):
        """
        Initialize the Gaussian Process model.
        
        Args:
            initialPoint: Reference point with assumed zero value
            theta: Kernel length scale parameter
            noise_level: Noise parameter for preference observations
        """
        self.listQueries = []  # List of queries and their responses
        self.K = np.zeros((2, 2))  # Covariance matrix for queries
        self.Kinv = np.zeros((2, 2))  # Inverse of covariance matrix
        self.fqmean = 0  # Posterior mean for queries
        self.theta = theta  # Kernel hyperparameter
        self.W = np.zeros((2, 2))  # Hessian at queries
        self.noise = noise_level  # Noise level for observations
        self.initialPoint = np.array(initialPoint)  # Reference point
        self.dim = len(self.initialPoint)  # Number of features

    def updateParameters(self, query, answer):
        """
        Update GP parameters after receiving a new preference observation.
        
        Args:
            query: Pair of points to compare [xa, xb]
            answer: Preference response (1 if xa > xb, -1 otherwise)
        """
        self.listQueries.append([query[0], query[1], answer])
        self.K = self.covK()
        # Add small diagonal term for numerical stability
        self.Kinv = inv(self.K + np.identity(2*len(self.listQueries))*1e-8)
        self.fqmean = self.meanmode()
        self.W = self.hessian()

    def objectiveEntropy(self, x):
        """
        Compute the objective function (entropy) for a query [xa, xb].
        This function is maximized to find the best query.
        
        Args:
            x: Concatenated vector of xa and xb points
            
        Returns:
            float: Information gain (entropy) for the query
        """
        xa = x[:self.dim]
        xb = x[self.dim:]

        matCov = self.postcov(xa, xb)
        mua, mub = self.postmean(xa, xb)
        sigmap = np.sqrt(np.pi * np.log(2) / 2) * self.noise

        # Compute entropy terms
        result1 = h(phi((mua - mub) / (np.sqrt(2*self.noise**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))))
        result2 = sigmap * 1 / (np.sqrt(sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1])) * np.exp(
            -0.5 * (mua - mub)**2 / (sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))

        return result1 - result2

    def kernel(self, xa, xb):
        """
        Compute the kernel function between two points.
        Uses a modified RBF kernel that accounts for the reference point.
        
        Args:
            xa: First point
            xb: Second point
            
        Returns:
            float: Kernel value
        """
        ker = (1 * (np.exp(-self.theta * np.linalg.norm(np.array(xa) - np.array(xb)) ** 2)) - 
               np.exp(-self.theta * np.linalg.norm(xa-self.initialPoint)**2) * 
               np.exp(-self.theta * np.linalg.norm(xb-self.initialPoint)**2))
        return ker

    def meanmode(self):
        """
        Find the posterior means for all queries using MAP estimation.
        
        Returns:
            array: Posterior means for all queries
        """
        n = len(self.listQueries)
        Kinv = self.Kinv
        listResults = np.array([q[2] for q in self.listQueries])

        def logposterior(f):
            """Compute the log posterior probability."""
            fodd = f[1::2]
            feven = f[::2]
            fint = 1/self.noise * (feven-fodd)
            res = np.multiply(fint, listResults)
            res = res.astype(dtype=np.float64)
            res = norm.cdf(res)
            res = np.log(res)
            res = np.sum(res)
            ftransp = f.reshape(-1, 1)
            return -1 * (res - 0.5 * np.matmul(f, np.matmul(Kinv, ftransp)))

        def gradientlog(f):
            """Compute the gradient of the log posterior."""
            grad = np.zeros(2*len(self.listQueries))
            for i in range(len(self.listQueries)):
                signe = self.listQueries[i][2]
                grad[2*i] = (self.listQueries[i][2] * 
                           (phip(signe*1/self.noise*(f[2*i]-f[2*i+1]))*1/self.noise) /
                           phi(signe*1/self.noise*(f[2*i]-f[2*i+1])))
                grad[2*i+1] = (self.listQueries[i][2] * 
                             (-phip(signe*1/self.noise*(f[2*i]-f[2*i+1]))*1/self.noise) /
                             phi(signe*1/self.noise*(f[2*i]-f[2*i+1])))
            grad = grad - f@Kinv
            return -grad

        x0 = np.zeros(2*n)
        return minimize(logposterior, x0=x0, jac=gradientlog).x

    def hessian(self):
        """
        Compute the Hessian matrix for all queries.
        
        Returns:
            array: Hessian matrix
        """
        n = len(self.listQueries)
        W = np.zeros((2*n, 2*n))
        for i in range(n):
            dif = self.listQueries[i][2] * 1/self.noise * (self.fqmean[2*i]-self.fqmean[2*i+1])
            W[2*i][2*i] = -(1/self.noise**2) * (phipp(dif)*phi(dif)-phip(dif)**2)/(phi(dif)**2)
            W[2*i+1][2*i] = -W[2*i][2*i]
            W[2*i][2*i+1] = -W[2*i][2*i]
            W[2*i+1][2*i+1] = W[2*i][2*i]
        return W

    def kt(self, xa, xb):
        """
        Compute covariance between query points and all previous queries.
        
        Args:
            xa: First point
            xb: Second point
            
        Returns:
            array: Covariance matrix
        """
        n = len(self.listQueries)
        return np.array([[self.kernel(xa, self.listQueries[i][j]) for i in range(n) for j in range(2)],
                        [self.kernel(xb, self.listQueries[i][j]) for i in range(n) for j in range(2)]])

    def covK(self):
        """
        Compute the full covariance matrix for all queries.
        
        Returns:
            array: Covariance matrix
        """
        n = len(self.listQueries)
        return np.array([[self.kernel(self.listQueries[i][j], self.listQueries[l][m]) 
                         for l in range(n) for m in range(2)] 
                        for i in range(n) for j in range(2)])

    def postmean(self, xa, xb):
        """
        Compute posterior mean for two points.
        
        Args:
            xa: First point
            xb: Second point
            
        Returns:
            tuple: Posterior means for both points
        """
        kt = self.kt(xa, xb)
        return np.matmul(kt, np.matmul(self.Kinv, self.fqmean))

    def cov1pt(self, x):
        """Compute variance for a single point."""
        return self.postcov(x, 0)[0][0]

    def mean1pt(self, x):
        """Compute mean for a single point."""
        return self.postmean(x, 0)[0]

    def postcov(self, xa, xb):
        """
        Compute posterior covariance matrix for two points.
        
        Args:
            xa: First point
            xb: Second point
            
        Returns:
            array: Posterior covariance matrix
        """
        n = len(self.listQueries)
        Kt = np.array([[self.kernel(xa, xa), self.kernel(xa, xb)],
                       [self.kernel(xb, xa), self.kernel(xb, xb)]])
        kt = self.kt(xa, xb)
        W = self.W
        K = self.K
        post_cov = Kt - kt @ inv(np.identity(2*n) + np.matmul(W, K)) @ W @ np.transpose(kt)
        return post_cov
        