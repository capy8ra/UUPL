from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
from util import *


class GaussianProcess:
    def __init__(self, initialPoint=0, theta=0.1, noise_level=0.1):
        self.listQueries =[]
        self.K = np.zeros((2,2))
        self.Kinv = np.zeros((2, 2))
        self.fqmean = 0
        self.theta = theta
        self.W = np.zeros((2,2))
        self.noise = noise_level
        self.initialPoint = np.array(initialPoint)
        self.dim = len(self.initialPoint)
        self.pref_dict = {}
        self.uncertainty_level = 0
        self.uncertainty_sigma_dict = {1: 0.01, 2: 0.66, 3: 1.7, 4: 3.35, 5: 9.0}

    def updateParameters(self, query, answer, uncertainty, pref_dict):
        self.listQueries.append([query[0],query[1],answer, uncertainty])
        self.uncertainty_level = uncertainty
        self.K = self.covK()
        self.Kinv = inv(self.K+np.identity(2*len(self.listQueries))*1e-8)
        self.fqmean = self.meanmode()
        self.W = self.hessian()
        self.pref_dict = pref_dict


    def objectiveEntropy(self, x): #Compute the objective function (entropy) for a query [xa,xb]
        xa = x[:self.dim]
        xb = x[self.dim:]

        matCov = self.postcov(xa, xb)
        mua, mub = self.postmean(xa, xb)
        sigmap = np.sqrt(np.pi * np.log(2) / 2)*self.noise

        result1 = h(
            phi((mua - mub) / (np.sqrt(2*self.noise**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))))
        result2 = sigmap * 1 / (np.sqrt(sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1])) * np.exp(
            -0.5 * (mua - mub)**2  / (sigmap ** 2  + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))

        return result1 - result2
    
    def GMM(self, xa, xb):
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

    
    def kernel(self, xa, xb):
        ker = 1*(np.exp(-self.theta*np.linalg.norm(np.array(xa) - np.array(xb)) ** 2))
        try:
            ker = ker[0]
        except:
            pass
        if ker < 0:
            print("You can not have a negative kernel!")
            exit()
        # print("ker:", ker)
        return ker
    
    def batch_kernel(self, xa, xb):
        num = xa.shape[0]
        xb = np.repeat(np.array(xb).reshape(1, -1), num, axis=0)
        ker = np.exp(-self.theta*np.linalg.norm(np.array(xa) - np.array(xb), axis=1) ** 2)
        return ker

    def meanmode(self): #find the posterior means for the queries
        n = len(self.listQueries)
        Kinv = self.Kinv
        # print(Kinv)
        listResults = np.array([q[2] for q in self.listQueries])
        sigmas = np.array([self.uncertainty_sigma_dict[q[3]] for q in self.listQueries])
        def logposterior(f):
            fodd  = f[1::2]
            feven = f[::2]
            fint = 1/self.noise*(feven-fodd)
            res = np.multiply(fint, listResults)
            res = res.astype(dtype = np.float64)
            res = norm.cdf(res, scale=sigmas[:n])
            res[res == 0] = 1e-100
            res = np.log(res)
            res = np.sum(res)
            ftransp = f.reshape(-1,1)
            ret = -1*(res- 0.5 * np.matmul(f, np.matmul(Kinv, ftransp)))
            # print(">>> logposterior:", ret)
            return ret


        def gradientlog(f):
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
            # print(">>> grad:", grad)
            return -grad
        x0 = np.zeros(2*n)
        ret = minimize(logposterior, x0=x0, jac=gradientlog).x
        # print(">>> return value:", ret)
        return ret


    def hessian(self):
        n = len(self.listQueries)
        W = np.zeros((2*n,2*n))
        for i in range(n):
            dif = self.listQueries[i][2]*1/self.noise*(self.fqmean[2*i]-self.fqmean[2*i+1])
            W[2*i][2*i] = -(1/self.noise**2)*(phipp(dif)*phi(dif)-phip(dif)**2)/(phi(dif)**2)
            W[2*i+1][2*i] = -W[2*i][2*i]
            W[2*i][2*i+1] = -W[2*i][2*i]
            W[2*i+1][2*i+1] = W[2*i][2*i]
        return W


    def kt(self, xa, xb, eval=False):  #covariance between xa,xb and our queries
        n = len(self.listQueries)
        if eval:
            return np.array([self.batch_kernel(xa,self.listQueries[i][j])for i in range(n) for j in range(2)])
        else:
            return np.array([[self.kernel(xa,self.listQueries[i][j])for i in range(n) for j in range(2)], [self.kernel(xb,self.listQueries[i][j])for i in range(n) for j in range(2)]])

    def covK(self): #covariance matrix for all of our queries
        n= len(self.listQueries)
        return np.array([[self.kernel(self.listQueries[i][j], self.listQueries[l][m]) for l in range(n) for m in range(2)] for i in range(n) for j in range(2)])

    def postmean(self, xa, xb, eval=False): #mean vector for two points xa and xb
        # print(xa.shape)
        kt = self.kt(xa,xb, eval=eval)
        if eval == True:
            kt = kt.T
        return np.matmul(kt, np.matmul(self.Kinv,self.fqmean))

    def cov1pt(self,x): #variance for 1 point
        return self.postcov(x,0)[0][0]

    def mean1pt(self,x,eval=False):
        # print(x.shape)
        if eval:
            return self.postmean(x,0, eval=eval)
        else:
            return self.postmean(x,0, eval=eval)[0]

    def postcov(self, xa, xb): #posterior covariance matrix for two points
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
