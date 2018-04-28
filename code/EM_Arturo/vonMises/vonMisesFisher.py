import numpy as np
from scipy.special import ive,iv
from scipy import pi

def get_phi(X,pz_x,K,N,p):
    
    pi = np.sum(pz_x,0)/N
    mu = np.zeros((p,K))
    r = np.zeros((p,K))
    for k in range(K):
        r[:,k] = np.sum(X.T*pz_x[:,k],1)
        mu[:,k] = r[:,k]/np.linalg.norm(r[:,k])
        
    kappa = np.zeros(K)
    for k in range(K):
        R = np.linalg.norm(r[:,k])/(np.sum(pz_x[:,k]))
        kappa[k] = (R*p - np.power(R,3))/(1-np.power(R,2))
        if (1-np.power(R,2))==0:
            print "WARNING: divided by 0"
            #import IPython
            #IPython.embed()

    return pi,mu,kappa
    
def cp(p,kappa):
    num = np.power(kappa,float(p)/2-1)
    den= np.power(2*pi,float(p)/2)*iv(float(p)/2-1,kappa)
    return num/den

def cp_log(p,kappa):
    p = float(p)
    cp = (p/2-1)*np.log(kappa)
    cp += -(p/2)*np.log(2*pi)-np.log(ive(p/2-1,kappa))-kappa
    return cp

def pdf_log(x,mu,kappa):
    shape = x.shape
    if len(shape) == 1:
        p = shape[0]
    else: p = shape[1]
    return cp_log(p,kappa) + np.dot(x,mu)*kappa

def pdf(x,mu,kappa):
    shape = x.shape
    if len(shape) == 1:
        p = shape[0]
    else: p = shape[1]
    ev = cp(p,kappa)*np.exp(np.dot(x,mu)*kappa)
    return ev

def MLestimator(X):
    """Maxiumum likelihood estimator for a single vonMises"""
    N,p = X.shape
    sum_x = np.sum(X,0)
    norm_sum_x = np.linalg.norm(sum_x)
    mu = sum_x/norm_sum_x

    R = norm_sum_x/N
    kappa0 = (R*(p-np.power(R,2)))/(1-np.power(R,2))
    A = iv(float(p)/2,kappa0)/iv(float(p)/2-1,kappa0)
    kappa1 = kappa0 - (A - R)/(1-np.power(A,2)-A*float(p-1)/kappa0)
    return mu,kappa1

def init_params(K,p):
    """Initilze the paramters for a moVMF
    Input:
        K: number of clusters
        p: number of dimensions
    """
    pi = np.ones(K)/K
    mu = np.random.rand(p,K)
    for i in range(K):
        mu[:,i] = mu[:,i]/np.linalg.norm(mu[:,i])
    mu[:,0] = mu[:,0]*(-1)
    kappa = np.random.rand(K)+2
    return pi, mu, kappa
