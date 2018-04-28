import numpy as np
import pandas as pd
import vonMisesFisher as vMF

def EM(X,K, maxit = 100, verbose = True):
    """perform Expectation maximization with K clusters"""
    N,p = X.shape
    phi = vMF.init_params(K,p)
    it = 0; tol = 1000
    llold = loglikelihood_log(X,phi)
    ll = [llold]
    #EM algorithm starts
    while tol >10e-5 and it < maxit:
        # E-step
        pz_x = get_respons_log(X,phi,K)
        #M-step
        phi = vMF.get_phi(X,pz_x,K,N,p)        
        #save data and update ll
        llnew = loglikelihood_log(X,phi)
        ll.append(llnew)
        tol = abs(llold - llnew)
        llold = llnew
        if verbose==True: 
            print loglikelihood_log(X,phi)
        it +=1
    return phi,ll

def severalEM(X,K,Ninit = 5, maxit = 100, verbose = False):
    """Run several EM algorithms and return the one with highest logkilelihood"""
    from math import isnan
    best_ll = -10000; best_phi = None 
    for i in range(Ninit):
        new_phi,new_ll = EM(X,K,maxit,verbose)
        if isnan(new_ll[-1])==False and new_ll[-1]>best_ll:
            best_ll = new_ll[-1]
            best_phi = new_phi
    return best_phi, best_ll

def vonCV_EM(X,K_list,conditions,output_path='./',Ninits = 5):
    """cross validation with LOO strategy"""
    num_conditions,num_subjects = X.shape
    df_list = []
    for i in range(num_conditions):
        condition = X[i,:]
        df = pd.DataFrame(index = K_list, 
                          columns=['train','train_std','LOO','LOO_std'])
        for k in K_list:
            test_ll = []; train_ll = []
            for j in range(num_subjects):
                print "==============================================="
                print"=======Condition: {}, num_K: {}, subject: {}==============".format(i,k,j)
                print"================================================"
                #Splitting data in train-validation
                LOO, rest = condition[j], np.delete(condition,j)
                rest_np = rest[0]
                for x in rest[1:]:
                    rest_np = np.concatenate([rest_np,x])
                #Perform EM over all subjects except one
                phi,new_train_ll = severalEM(rest_np,k,Ninits)
                #Test the model over the last subject
                new_test_ll = loglikelihood_log(LOO,phi)
                test_ll.append(new_test_ll)
                train_ll.append(new_train_ll/(num_subjects-1))
            df.set_value(k,'LOO',mean(test_ll))
            df.set_value(k,'train',mean(train_ll))
            df.set_value(k,'LOO_std',std(test_ll))
            df.set_value(k,'train_std',std(train_ll))
            df.to_csv(output_path+'vonM_'+conditions[i]+'.csv',sep=',')
        df_list.append(df)
    return df_list

def mean(alist):
    return np.array(alist).mean()

def std(alist):
    return np.std(np.array(alist))

def loglikelihood_log(X,phi):
    """ incomplete loglikehood P(X|phi). It is more stable than loglikelihood()
    Input:
        X: narray, N(samples)*p(dimensions)
        phi:tuple(pi,mu,kappas)mixing coefficients and distribution parameters
    Output:
        ll: P(X|phi)
    """
    pi = phi[0]; mu = phi[1]; kappa = phi[2]
    N,p = X.shape
    K = len(pi)
    pz_x = np.zeros(N)
    for k in range(K):
        pz_x += np.exp(vMF.pdf_log(X,mu[:,k],kappa[k]))*pi[k]
    ll = np.sum(np.log(pz_x))
    #debugging
    from math import isnan
    if isnan(ll)==True:
        print "CLUSTER CRUSHED: SINGULARITY"
    return ll

def get_respons_log (X,phi,K):
    """ get the responsabilities. Internally it use logarithmic scaling
    so it is more robust than get_respons()
    Input:
        X: narray, N(samples)*p(dimensions)
        phi:tuple(pi,mu,kappas)mixing coefficients and distribution parameters
        K: number of clusters
    Output:
        pz_x: probability of sample xn to belong to each cluster k
    """
    shape = X.shape
    if len(shape) == 1:
        p = shape[0]; N = 1
    else: p = shape[1]; N = shape[0]
    pi = phi[0]; mu = phi[1]; kappa = phi[2] 
    px_z = np.zeros((N,K))
    for k in range(K):
        px_z[:,k] = np.exp(vMF.pdf_log(X,mu[:,k],kappa[k]))
    pz_x_unormalize = px_z*pi
    normalizer = np.sum(pz_x_unormalize,1)
    pz_x = pz_x_unormalize.T/normalizer
    return pz_x.T

def get_respons (X,phi,K):
    """ get the responsabilities
    Input:
        X: narray, N(samples)*p(dimensions)
        phi:tuple(pi,mu,kappas)  parameters from distributions
        K: number of clusters
    Output:
        pz_x: probability of sample xn to belong to each cluster k
    """
    shape = X.shape
    if len(shape) == 1:
        p = shape[0]; N = 1
    else: p = shape[1]; N = shape[0]
    pi = phi[0]; mu = phi[1]; kappa = phi[2] 
    px_z = np.zeros((N,K))
    for k in range(K):
        px_z[:,k] = vMF.pdf(X,mu[:,k],kappa[k])
    pz_x_unormalize = px_z*pi
    normalizer = np.sum(pz_x_unormalize,1)
    pz_x = pz_x_unormalize.T/normalizer
    return pz_x.T

def loglikelihood(X,phi):
    """ incomplete loglikehood P(X|phi). 
    Input:
        X: narray, N(samples)*p(dimensions)
        phi:tuple(pi,mu,kappas)mixing coefficients and distribution parameters
    Output:
        ll: P(X|phi)
    """
    pi = phi[0]; mu = phi[1]; kappa = phi[2]
    N,p = X.shape
    K = len(pi)
    pz_x = np.zeros(N)
    for k in range(K):
        pz_x += vMF.pdf(X,mu[:,k],kappa[k])*pi[k]
    ll = np.sum(np.log(pz_x))
    return ll

