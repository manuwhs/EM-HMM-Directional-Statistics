# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:47:19 2017

@author: montoya
"""

import import_folders
from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.optimize import newton
import numpy as np
import utilities_lib as ul
import HMM_libfunc2 as HMMl

def get_cp(Ndim, kappa):
    gammaValue = gamma(float(Ndim)/2)
    M = hyp1f1(0.5, float(Ndim)/2, kappa)   # Confluent hypergeometric function 1F1(a, b; x)
    cp = gammaValue / (np.power(2*np.pi, float(Ndim)/2)* M)
    return cp
    ## TODO: Make it one func
    #def get_K_cps_log(D,kappas):
    #    # This function will compute the constants for several clusters
    #    cp_logs = []
    #    K = kappas.shape[1]
    #    for k in range(K):
    #        cp_logs.append(difu.get_cp_log(D,kappas[:,k]))
def get_cp_log(Ndim, kappa):
    gammaValue_log = np.log(gamma(float(Ndim)/2))
    M_log = kummer_log(0.5, float(Ndim)/2, kappa)   # Confluent hypergeometric function 1F1(a, b; x)
    cp_log = gammaValue_log - (np.log(2*np.pi) *(float(Ndim)/2) + M_log)

    return cp_log
    
def check_Kummer(Ndim, kappa):
    # This functions checks if the Kummer function will go to inf
    # Returns 1 if Kummer is stable, 0 if unstable
    f = hyp1f1(0.5, float(Ndim)/2, kappa) 
    if (np.isinf(f) == False):
        return 1
    else:
        return 0
        
def kummer_log(a,b,x):
    ## First try using the funcion in the library
    f = hyp1f1(a,b,x)
    if (np.isinf(f) == False):
        return np.log(f)
    else:
        print "The kummer function would give too high value"
    # Default tolerance is tol = 1e-10.  Feel free to change this as needed.
    tol = 1e-10;
    log_tol = np.log(tol)
    # Estimates the value by summing powers of the generalized hypergeometric
    # series:
    #      sum(n=0-->Inf)[(a)_n*x^n/{(b)_n*n!}
    # until the specified tolerance is acheived.
    
    log_term = np.log(x) + np.log(a) - np.log(b)
#    f_log =  HMMl.sum_logs([0, log_term])
    
    n = 1;
    an = a;
    bn = b;
    nmin = 5;
    
    terms_list = []
    
    terms_list.extend([0,log_term])
    d = 0
    while((n < nmin) or (log_term > log_tol)):
      # We increase the n in 10 by 10 reduce overheading of  while
      n = n + d;
#      print "puto n %i"%(n)
#      print f_log
      an = an + d;
      bn = bn + d;
      
      d = 1
#      term = (x*term*an)/(bn*n);
      log_term1 = np.log(x) + log_term  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term2 = np.log(x) + log_term1  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term3 = np.log(x) + log_term2  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term4 = np.log(x) + log_term3  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term = np.log(x) + log_term4  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
  
      terms_list.extend([log_term1,log_term2,log_term3,log_term4,log_term] )
      
      if(n > 10000):
          print "Too many terms in kummer"
          break
    f_log = HMMl.sum_logs(terms_list);
#    print n
    return f_log

def Watson_pdf (alpha, mu, kappa, cp = None):
    # Watson pdf for a 
    # mu: [mu0 mu1 mu...] p-1 dimesion angles in radians
    # kappa: Dispersion value
    # alpha: Vector of angles that we want to know the probability
####
    # Just make sure the matrixes are aligned
    mu = np.array(mu)
    mu = mu.flatten().reshape(mu.size,1)
    
    alpha = np.array(alpha)
    alpha = alpha.reshape(mu.size,alpha.size/mu.size)
    
    Ndim = mu.size #+ 1  ??
    
    # If we indicate cp, we do not compute it
    if (cp == None):
        cp = get_cp(Ndim, kappa)
#        print "GRGRGR"
#    print np.dot(mu.T, alpha)
    
    aux1 = np.dot(mu.T, alpha)
#    aux2 = 0
#    for i in range(mu.size):
#        aux2 = aux2 + mu[i]*alpha[i]
#    print alpha
    
#    if (kappa < 0):
#        print "Warning: Kappa < 0"
        
    pdf = cp * np.exp(kappa * np.power(aux1,2))
    
    if (pdf.size == 1): # Turn it into a single number if appropiate
        pdf = float(pdf)
    return pdf
    
      
def Watson_pdf_log (alpha, mu, kappa, cp_log = None):
    # Compute this in case that the probability is too high or low for just one sample
    # cp is ok, we can calculate normaly and then put it log
    # cp goes to very high if low dimensions and high kappa
    
    # If we indicate cp_log  we do not compute it.
    
    # Watson pdf for a 
    # mu: [mu0 mu1 mu...] p-1 dimesion angles in radians
    # kappa: Dispersion value
    # alpha: Vector of angles that we want to know the probability
####
    # Just make sure the matrixes are aligned

    mu = np.array(mu)
    mu = mu.flatten().reshape(mu.size,1)
    
    alpha = np.array(alpha)
    alpha = alpha.reshape(mu.size,alpha.size/mu.size)
    
    Ndim = mu.size #+ 1  ??
    
    if (type(cp_log) == type(None)):
        cp_log = get_cp_log(Ndim, kappa)
        
#    print np.dot(mu.T, alpha)
    
    aux1 = np.dot(mu.T, alpha)
#    aux2 = 0
#    for i in range(mu.size):
#        aux2 = aux2 + mu[i]*alpha[i]
#    print alpha

    log_pdf = cp_log + (kappa * np.power(aux1,2))
    
    if (log_pdf.size == 1): # Turn it into a single number if appropiate
        log_pdf = float(log_pdf)
    return log_pdf


def Watson_K_pdf_log (alpha, mus, kappas, cps_log = None):
    # Extension of Watson_pdf_log in which we also accept several clusters
    # We have to be more restrict in this case and the parameters must be:
    # alpha(D,Nsamples)  mu(D,K) kappa(K) cp_log(K)
    # The result is (Nsamples, K)

    Ndim, Nsam = alpha.shape
    Ndim2, K = mus.shape
    
    if (type(cps_log) == type(None)):
        cps_log = get_cp_log(Ndim, kappas)
        
    kappas = np.array(kappas)
    kappas = kappas.reshape(kappas.size,1)
    cps_log = np.array(cps_log)
    cps_log = cps_log.reshape(cps_log.size,1)
    
    aux1 = np.dot(mus.T, alpha)
#    aux2 = 0
#    for i in range(mu.size):
#        aux2 = aux2 + mu[i]*alpha[i]
#    print alpha
    log_pdf = cps_log + (kappas * np.power(aux1,2))
    return log_pdf.T
    
def get_MLmean(X, S = None):
    n,d = X.shape
    # Check if we are given the S
    # Maybe not used in the end, it has to be done together with Kappa
    # To check if we need to return mu1 or mup
    if (type(S) == type(None)):
        S = np.dot(X.T,X)   # Correlation
    S = S/n             # Not really necesarry
    D,V = np.linalg.eig(S) # Obtain eigenvalues D and vectors V
    
    if (D[0] == D[1]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    return V[:,0]

def get_Weighted_MLMean(rk,X, Sk = None):
    
    n,d = X.shape
    # Check if we are given the S
    if (type(Sk) == type(None)):
        Sk = np.dot(X.T,rk*X)   # Correlation # We weight the samples by the cluster responsabilities r[:,k]
        # Correlation
    
    Sk = Sk/n             # Not really necesarry
    D,V = np.linalg.eig(Sk) # Obtain eigenvalues D and vectors V
    
    max_d = np.argmax(D)
    if (D[max_d] == D[max_d]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    return V[:,max_d]

def get_kappaNewton(k, args):  # The r is computed outsite
    Ndim = args[0]
    r = args[1]
    
    a = 0.5
    c = float(Ndim)/2
    
    M = np.exp(kummer_log(a, c,k))
    Mplus = np.exp(kummer_log(a + 1, c +1,k))
    dM = (a/c)*Mplus 

    g = dM/M
#    kummer = 
#    print Ndim, k, r
    return g - r

def Newton_kappa(kappa0,Ndim,r, Ninter = 10):
    kappa = kappa0
    a = 0.5
    c = float(Ndim)/2
    for i in range(Ninter):
        
        M = np.exp(kummer_log(a, c,kappa))
        Mplus = np.exp(kummer_log(a + 1, c +1,kappa))
        dM = (a/c)*Mplus 
#        dM = (a - c)*Mbplus/c + M
        g = dM/M
#        print g
        dg =  (1 - c/kappa)*g + (a/kappa) - g*g
        
        kappa = kappa - (g - r)/dg
        
#        print kappa
    return kappa
    
def get_MLkappa(mu,X, S = None):
    n,d = X.shape
    if (type(S) == type(None)):
        S = np.dot(X.T,X)   # Correlation
        
    S = S/n
    r = np.dot(mu.T,S).dot(mu)
#    print r
    
    a = 0.5
    c = float(d)/2
    
    # General aproximation
#    BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))
    
    # When r -> 1 
    BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
#    print BGG
    
#    BGG = Newton_kappa(BGG,d,r,Ninter = 5)
#    BGG = newton(get_kappaNewton, BGG, args=([d,r],))
#    print "STSHNWSRTNSRTNWRSTN"
    return BGG

def get_Watson_muKappa_ML(X):
    # This function obtains both efficiently and checking the sign and that
    n,d = X.shape
    a = 0.5
    c = float(d)/2
    
    S = np.dot(X.T,X)   # Correlation
    S = S/n             # Not really necesarry

    # Get eigenvalues to obtain the mu
    D,V = np.linalg.eig(S) # Obtain eigenvalues D and vectors V
    
    print D
    
    d_pos = np.argmax(D)
    d_min = np.argmin(D)
    ## We first assume it is positive if not we change the mu
    if (D[0] == D[1]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    if (D[-1] == D[-2]):
        print "Warning: Eigenvaluep = EigenValuep-1 in MLmean estimation"

    ## We solve the positive and the negative situations and output the one with
    ## the highest likelihood ? 

    mu_pos = V[:,d_pos]  # This is the vector with the highest lambda
    mu_neg = V[:,d_min]  # This is the vector with the lowest lambda
    
    r_pos = np.dot(mu_pos.T,S).dot(mu_pos)
    r_neg = np.dot(mu_neg.T,S).dot(mu_neg)
#    print r

    # General aproximation
    BGG_pos = (c*r_pos -a)/(r_pos*(1-r_pos)) + r_pos/(2*c*(1-r_pos))
#    kappa_pos = BGG_pos
    kappa_pos = newton(get_kappaNewton, BGG_pos, args=([d,r_pos],))

    BGG_neg = (c*r_neg -a)/(r_neg*(1-r_neg)) + r_neg/(2*c*(1-r_neg))
#    kappa_neg = BGG_neg
    kappa_neg = newton(get_kappaNewton, BGG_neg, args=([d,r_neg],))
    
    likelihood_pos = np.sum(np.exp(Watson_pdf_log(X.T,mu_pos,kappa_pos)))
    likelihood_neg = np.sum(np.exp(Watson_pdf_log(X.T,mu_neg,kappa_neg)))
    
    print likelihood_pos, likelihood_neg
    if (likelihood_pos > likelihood_neg):
        kappa = kappa_pos
        mu = mu_pos
    else:
        kappa = kappa_neg
        mu = mu_neg
    return mu, kappa

def get_Watson_Wighted_muKappa_ML(X, rk):
    # This function obtains both efficiently and checking the sign and that
    n,d = X.shape
    a = 0.5
    c = float(d)/2
#    print (X*rk).shape
    
    Sk = np.dot(X.T,X*rk)   # Correlation
    Sk = Sk/np.sum(rk)            # Not really necesarry

    # Get eigenvalues to obtain the mu
    D,V = np.linalg.eig(Sk) # Obtain eigenvalues D and vectors V
    
#    print D
    
    d_pos = np.argmax(D)
    d_min = np.argmin(D)
    ## We first assume it is positive if not we change the mu
    if (D[0] == D[1]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    if (D[-1] == D[-2]):
        print "Warning: Eigenvaluep = EigenValuep-1 in MLmean estimation"

    ## We solve the positive and the negative situations and output the one with
    ## the highest likelihood ? 

    mu_pos = V[:,d_pos]  # This is the vector with the highest lambda
    mu_neg = V[:,d_min]  # This is the vector with the lowest lambda
    
    r_pos = np.dot(mu_pos.T,Sk).dot(mu_pos)
    r_neg = np.dot(mu_neg.T,Sk).dot(mu_neg)
#    print r

    # General aproximation
    BGG_pos = (c*r_pos -a)/(r_pos*(1-r_pos)) + r_pos/(2*c*(1-r_pos))
#    kappa_pos = BGG_pos
    kappa_pos = newton(get_kappaNewton, BGG_pos, args=([d,r_pos],))

    BGG_neg = (c*r_neg -a)/(r_neg*(1-r_neg)) + r_neg/(2*c*(1-r_neg))
#    kappa_neg = BGG_neg
    kappa_neg = newton(get_kappaNewton, BGG_neg, args=([d,r_neg],))
    
#    likelihood_pos = np.sum(Watson_pdf_log(X.T,mu_pos,kappa_pos))
#    likelihood_neg = np.sum(Watson_pdf_log(X.T,mu_neg,kappa_neg))

#    likelihood_pos = np.sum(np.exp(Watson_pdf_log(X.T,mu_pos,kappa_pos))*rk.T)
#    likelihood_neg = np.sum(np.exp(Watson_pdf_log(X.T,mu_neg,kappa_neg))*rk.T)

    # The maximum weighted likelihood estimator
    likelihood_pos = np.sum(Watson_pdf_log(X.T,mu_pos,kappa_pos)*rk.T)
    likelihood_neg = np.sum(Watson_pdf_log(X.T,mu_neg,kappa_neg)*rk.T)

 
    print likelihood_pos, likelihood_neg
    if (likelihood_pos > likelihood_neg):
        kappa = kappa_pos
        mu = mu_pos
    else:
        kappa = kappa_neg
        mu = mu_neg
    return mu, kappa
    
def get_Weighted_MLkappa(rk, mu,X, Sk = None):
    n,d = X.shape

    if (type(Sk) == type(None)):
        Sk = np.dot(X.T,rk*X)   # Correlation # We weight the samples by the cluster responsabilities r[:,k]
        # Correlation
#    print r
    Sk = Sk/(np.sum(rk))
    
    r = np.dot(mu.T,Sk).dot(mu)
    a = 0.5
    c = float(d)/2
    
    # General aproximation
    BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))
    
    # When r -> 1 
#    BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
    # TODO: In some examples this does not converge in the scipy method..
#    BGG = newton(get_kappaNewton, BGG, args=([d,r],))
    BGG = Newton_kappa(BGG,d,r,Ninter = 30)
    return BGG

def randWatsonMeanDir(N, kappa, p):
    # Generate samples of the unidimensional watson (0,..,0,1)
    # p is for the scalling of the cp
    ## For some reason this is multiplied by exactly p-1
    compute_leftBound =1
#    kappa = kappa/(p-1)
    print p
    normali = np.exp(get_cp_log(p,kappa) - get_cp_log(2,kappa)) ## TODO: It is a 2 right ?
    print "Normalization %f" % normali
    
    if (compute_leftBound):
        min_thresh = 1/(float(5)) #
        ### THIS IS JUST TO GET LEFT BOUNDARY AND PDF BOUNDATY ?
        step = 0.00001
        xx = np.arange(0, 1+step, step)
        xx = np.array(xx) * 2 * np.pi
        xx = np.array([np.cos(xx ), np.sin(xx)])
        
#        print xx.shape
#        xx = xx.T

        mux = np.array([1,0])
        
        # Get a grid of the univariate Watson
        yy = np.exp(Watson_pdf_log(xx, mux, kappa)) * normali

        # Get the cumulative distribution
        cumyy = yy.cumsum(axis=0)*(xx[1]-xx[0])
        
        print np.max(yy)
#        print yy
        # Take care of the Boundaries
#        leftBound = xx[np.ndarray.flatten(np.asarray((np.nonzero(cumyy>min_thresh/2.0))))][0]
        leftBound = 0
#        print leftBound
    else:
        leftBound = 0.000
    leftBound = 0.000
#    print leftBound
    # Get the maximum probability of one of the samples
    if (kappa > 0):
        M =  np.exp(get_cp_log(p,kappa)) * np.exp(kappa)
        print "Kappa positive, M: %f" % M
    else:
        M = np.exp(get_cp_log(p,kappa)) * 1 #np.exp(-kappa)
        print "Kappa negative, M: %f" % M
#        print M
        
    t = np.zeros(int(N))
#    leftBound = 0.0
    # For every sample
    for i in range (0, int(N)):
        while 1:
            # TODO: Obtain a lot of samples first from np.random.uniform() snd
            # do all the process vectorialy.
        
            # Get uniform distribution in the limits
            x = np.random.uniform(0.0, 1.0)*(1-leftBound)+leftBound
            x = [x, 0]
#            x = np.array(x) * 2 * np.pi
#            x = np.array([np.cos(x), np.sin(x)])
            mux = np.array([1,0])
            # Compute the pdf of the random sample
            h = np.exp(Watson_pdf_log(x, mux, kappa)) * normali
#            print h
            # If the sample pdf is bigger than the M we stop
            draw = np.random.uniform(0.0, 1.0)*M* (0.999999)
            
            ## TODO: Here is the shit to avoid the problem of that the maximum does not happen
            if draw <=h:
                break

        if np.random.uniform(0.0, 1.0)>0.5:
#            print 2
            t[i] = float(x[0])
        else:
            t[i] = -float(x[0])
    return np.asarray(t)

def randUniformSphere(N, p):
    # Generate N random vectors in the 1 sphere of dimension p
    randNorm = np.random.normal(0, 1, size=[N, p])
    RandSphere = np.zeros((N, p))

    for r in range(0, N):
        RandSphere[r,] = np.divide(randNorm[r,], np.linalg.norm(randNorm[r,]))
    return RandSphere

def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:].T.copy()

def randWatson(N, mu, k):
    # Generates N samples of a Watson distribution 
    # with mu and kappa
#    muShape = np.shape(mu)
#    print muShape
#    p = muShape[0]
    p =  np.array(mu).size
    tmpMu = np.zeros(p)
    tmpMu[0] = 1
    
    # 
    t = randWatsonMeanDir(N, k, p)
#    print t.shape
    RandSphere = randUniformSphere(int(N), p - 1)

    t_m = np.tile(t, (p, 1)).transpose()
    tmpMu_m = np.tile(tmpMu, (N, 1))

    t_m2 = np.tile(((1 - t**2)**(0.5)), [p, 1]).transpose()
    RNDS = np.c_[np.zeros(int(N)), np.asarray(RandSphere)]

    RandWatson = t_m * tmpMu_m + t_m2*RNDS

    # Rotate the distribution to the right direction
    Otho = null(np.matrix(mu))

    Rot = np.c_[mu, Otho]
    RandWatson = (Rot * RandWatson.transpose()).conj()
    
    return np.array(RandWatson).T


def normalize_data(Xdata):
    tol = 0.0000001
    # Expects a matrix (Nsamples, Ndim) and normalizes the values
    Nsamples, Ndim = Xdata.shape
    Module = np.sqrt(np.sum(np.power(Xdata,2),1))
    Module = Module.reshape(Nsamples,1)
    # Check that the modulus is not 0
    Xdata = Xdata[np.where(Module > tol)[0],:]
    Xdata = np.divide(Xdata,Module)
    
    return Xdata


def draw_HMM_indexes(pi, A, Nchains = 10, Nsamples = 30):
    # If Nsamples is a number then all the chains have the same length
    # If it is a list, then each one can have different length
    K = pi.size  # Number of clusters
    Chains_list = []
    
    Cluster_index = range(K)
    
    Nsamples = ul.fnp(Nsamples)
    if(Nsamples.size == 1):  # If we only have one sample 
        Nsamples = [int(Nsamples)]*Nchains
        
    for nc in range(Nchains):
        Chains_list.append([])
        sample_indx = np.random.choice(Cluster_index, 1, p = pi)
        Chains_list[nc].append(int(sample_indx))
        
        for isam in range(1,Nsamples[nc]):
            # Draw a sample according to the previous state
            sample_indx = np.random.choice(Cluster_index, 1, 
                                           p = A[sample_indx,:].flatten())
            Chains_list[nc].append(int(sample_indx))
    
    return Chains_list

def draw_HMM_samples(Chains_list, Samples_clusters):
    # We take the indexes of the chain and then draw samples from a pregenerated set
    # Samples_clusters is a list where every element is an array of sampled of the
    # i-th cluster
    
    K = len(Samples_clusters)

    Nchains = len(Chains_list)
    HMM_chains = [];
    
    counter_Clusters = np.zeros((K,1))
    for nc in range(Nchains):
        Nsamples = len(Chains_list[nc])
        HMM_chains.append([])
        
        for isam in range(0,Nsamples):
            K_index = Chains_list[nc][isam]
            Sam_index = int(counter_Clusters[K_index])
  
            sample = Samples_clusters[K_index][Sam_index,:]
            counter_Clusters[K_index] = counter_Clusters[K_index] +1
            HMM_chains[nc].append(sample)
    
        HMM_chains[nc] = np.array(HMM_chains[nc])
    return HMM_chains
    