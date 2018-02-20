
from scipy.special import hyp1f1
from scipy.special import gamma
import numpy as np
import general_func as gf 

def get_responsabilityMatrix2(X,theta, pimix):
    mus = theta[0]
    kappas = theta[1]
    N, D = X.shape
    D,K = mus.shape
    r = np.zeros((N,K))
    
    for i in range(N):  # For every sample
        # For every  component
        for k in range (K):
            k_component_pdf = Wad.Watson_pdf(X[i,:], mus[:,k], kappas[:,k])
            
#                print pimix[:,k]
#                print k_component_pdf
            r[i,k] = pimix[:,k]*k_component_pdf
            
        # Normalize the probability of the sample being generated by the clusters
        Marginal_xi_probability = np.sum(r[i,:])
        r[i,:] = r[i,:]/Marginal_xi_probability
    return r


def get_responsabilityMatrix_log(X,theta, pimix): 
    mus = theta[0]
    kappas = theta[1]
    N, D = X.shape
    D,K = mus.shape
    r_log = np.zeros((N,K))
    
    # Truco del almendruco
    cp_logs = []
    for k in range(K):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,k]))
        
    r_log = np.zeros((N,K))
    
    # For every  component
    k_component_pdf = Wad.Watson_K_pdf_log(X[:,:].T, mus[:,:], kappas[:,:], cps_log = cp_logs)
    r_log = k_component_pdf  +  np.log(pimix[:,:]) 
    
    for i in range(N):  # For every sample
        # Normalize the probability of the sample being generated by the clusters
        Marginal_xi_probability = gf.sum_logs(r_log[i,:])
        r_log[i,:] = r_log[i,:]- Marginal_xi_probability

    return r_log


def get_responsabilityMatrix_log2(X,theta, pimix):
    mus = theta[0]
    kappas = theta[1]
    N, D = X.shape
    D,K = mus.shape
    r_log = np.zeros((N,K))
    
    # Truco del almendruco
    cp_logs = []
    for k in range(K):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,k]))
    
    def get_custersval(k):
        return Wad.Watson_pdf_log(X[i,:], mus[:,k], kappas[:,k], cp_log = cp_logs[k]) + np.log(pimix[:,k])

    krange = range(K)
    for i in range(N):  # For every sample
        # For every  component
        r_log[i,:] = np.array(map(get_custersval,krange)).flatten() 
        # Normalize the probability of the sample being generated by the clusters
        Marginal_xi_probability = gf.sum_logs(r_log[i,:])
        r_log[i,:] = r_log[i,:]- Marginal_xi_probability
        
    return r_log

def EMdecode(Xdata, theta, pimix):
    # This function will tell you for each point in the data set the most likely point that generated it
    n,d = Xdata.shape
    d,K = Xdata.shape
    
    r_log = get_responsabilityMatrix_log(Xdata,theta,pimix)
    
    Sel = np.argmax(r_log, axis = 1)
    
    return Sel

def get_EM_Incomloglike(theta,pimix,X):

    mus = theta[0]
    kappas = theta[1]
    print kappas
    N = X.shape[0] # Number of IDD samples
    K = kappas.size
    # Calculate log-likelihood
    ll = 0;
    for i in range (N):
        aux = 0;
        for k in range(K):
            k_component_pdf = distribution.pdf_log_K(X.T,theta , Cs_log = Cs_log)
            aux = aux +  pimix[:,k]*k_component_pdf

        ll = ll+ np.log(aux);
    
    return ll

def get_EM_Incomloglike_log(X, distribution, theta,pimix):

    N, D = X.shape
    K = len(theta)
    r_log = np.zeros((N,K))
    
    # We can precompute the normalization constants if we have the functionality !
    if (type(distribution.get_Cs_log) != type(None)):
        Cs_log = []
        for k in range(K):
            Cs_log.append(distribution.get_Cs_log(theta[k]))
    else:
        Cs_log = None
        
    ll = 0
 
    # For every  component
    k_component_pdf = Wad.Watson_K_pdf_log(X.T,theta , Cs_log = Cs_log)
    r_log = np.log(pimix[:,:]) + k_component_pdf
    
    ll = 0
    for i in range(N):  # For every sample
#        print "r_log"
#        print r_log[i,:]
        ll += gf.sum_logs(r_log[i,:])  # Marginalize clusters and product of samples probabilities!!
        # Normalize the probability of the sample being generated by the clusters
#        Marginal_xi_probability = gf.sum_logs(r_log[i,:])
#        r_log[i,:] = r_log[i,:]- Marginal_xi_probability
    return  ll

#    ll = 0
#    for i in range (N):
#        aux = [];
#        for k in range(K):
#            k_component_pdf = Wad.Watson_pdf_log(X[i,:], mus[:,k], kappas[:,k], cp_log = cp_logs[k])
#            aux.append( np.log(pimix[:,k]) + k_component_pdf)
#        ll = ll + gf.sum_logs(aux)

def get_EM_Incomloglike_byCluster_log(theta,pimix,X):
    # Gets the incomloglikelihood by clusters of all the samples

    mus = theta[0]
    kappas = theta[1]
    N = X.shape[0] # Number of IDD samples
    D = X.shape[1] # Number of dimesions of samples
    K = kappas.size # Number of clusters
    # Calculate log-likelihood
    # Truco del almendruco
    cp_logs = []
    for k in range(K):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,k]))

    k_component_pdf = Wad.Watson_K_pdf_log(X[:,:].T, mus[:,:], kappas[:,:], cps_log = cp_logs)
    r_log = np.log(pimix[:,:]) + k_component_pdf

    # Normalize probabilities first ?
    for i in range(N):  # For every sample
        Marginal_xi_probability = gf.sum_logs(r_log[i,:])
        r_log[i,:] = r_log[i,:] - Marginal_xi_probability
    

    clusters = np.argmax(r_log, axis = 1) # Implemented already
#    print clusters
    pi_estimated = []
    
    for i in range (K):
        pi_i = np.where(clusters == i)[0].size
        pi_estimated.append( np.array(pi_i)/float(N))
    
#    For ll in logll
#    
#        Logglikelihoods[clusters[i]] += ll

#    r_log = np.exp(r_log)
    ll = np.sum(r_log, axis = 0).reshape(1,K)

    # Mmm filter ?
#    ll = []
#    for i in range (K):
#        ll_i = np.sum(r_log[np.where(clusters == i)[0],i])
#        ll.append(ll_i)
#    ll = np.array(ll).reshape(1,K)
#    pi_estimated = np.array(pi_estimated).reshape(1,K)
#    ll = np.exp(ll)
#    print pi_estimated
    return  ll   #ll
    
def get_r_and_ll_old(X,theta, pimix):
    # Combined funciton to obtain the loglikelihood and r in one step
    mus = theta[0]
    kappas = theta[1]
    N, D = X.shape
    D,K = mus.shape
    r_log = np.zeros((N,K))
    
    # Truco del almendruco
    cp_logs = []
    for k in range(K):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,k]))
    
    ll = 0
    for i in range(N):  # For every sample
        # For every  component
        for k in range (K):
            
            k_component_pdf = Wad.Watson_pdf_log(X[i,:], mus[:,k], kappas[:,k], cp_log = cp_logs[k])
            r_log[i,k] = np.log(pimix[:,k]) + k_component_pdf
            
        ll += gf.sum_logs(r_log[i,:])
        # Normalize the probability of the sample being generated by the clusters
        Marginal_xi_probability = gf.sum_logs(r_log[i,:])
        r_log[i,:] = r_log[i,:]- Marginal_xi_probability
    return r_log, ll

def get_r_and_ll_old2(X,theta, pimix):
    # Combined funciton to obtain the loglikelihood and r in one step
    mus = theta[0]
    kappas = theta[1]
    N, D = X.shape
    D,K = mus.shape
    r_log = np.zeros((N,K))
    
    # Truco del almendruco
    cp_logs = []
    for k in range(K):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,k]))
    
    ll = 0
    for i in range(N):  # For every sample
        # For every  component
        k_component_pdf = Wad.Watson_K_pdf_log(X[[i],:].T, mus[:,:], kappas[:,:], cps_log = cp_logs)
        r_log[i,:] = np.log(pimix[:,:]) + k_component_pdf
            
        ll += gf.sum_logs(r_log[i,:])
        # Normalize the probability of the sample being generated by the clusters
        Marginal_xi_probability = gf.sum_logs(r_log[i,:])
        r_log[i,:] = r_log[i,:]- Marginal_xi_probability
    return r_log, ll



def get_r_and_ll(X,distribution,theta, pimix):
    # Combined funciton to obtain the loglikelihood and r in one step

    N, D = X.shape
    K = len(theta)
    r_log = np.zeros((N,K))
    
    # We can precompute the normalization constants if we have the functionality !
    if (type(distribution.get_Cs_log) != type(None)):
        Cs_log = []
        for k in range(K):
            Cs_log.append(distribution.get_Cs_log(theta[k]))
    else:
        Cs_log = None
        
    ll = 0

     # Compute the pdf for all samples and all clusters
    k_component_pdf = distribution.pdf_log_K(X.T,theta , Cs_log = Cs_log)
    r_log[:,:] = np.log(pimix[:,:]) + k_component_pdf
    
    for i in range(N):  # For every sample
        ll += gf.sum_logs(r_log[i,:])  # Marginalize clusters and product of samples probabilities!!
        # Normalize the probability of the sample being generated by the clusters
        Marginal_xi_probability = gf.sum_logs(r_log[i,:])
        r_log[i,:] = r_log[i,:]- Marginal_xi_probability
    return r_log, ll


def init_model_params(K,pi_init = None):
    # Here we will initialize the mixing coefficients parameters of the mixture model
    # MIXING COEFICIENTS
    # We set the mixing coefficients with uniform discrete distribution, this
    # way, the a priori probability of any vector to belong to any component is
    # the same.
    if (type(pi_init) == type(None)): # If not given an initialization
        pimix = np.ones((1,K));
        pimix = pimix*(1/float(K));
    else:
        pimix = np.array(pi_init).reshape(1,K)
        
    return pimix

def get_pimix(r):
    N,K = r.shape
    pimix = np.zeros((1,K))
    for k in range(K):
        pimix[:,k] = np.sum(r[:,k])/N;

    return pimix

def get_theta(X, r, distribution):
    """ This function aims to estimate the new theta values for all the clusters.
        For each cluster it will call the estimation function "distribution.theta_estimator(X, rk)".
        If it fails, it should create a RuntimeExeption, which is handled here by setting the parameters to None.
        This will be handled later.
    
    """
    # Cluster by Cluster it will estimate them, if it cannot, then it 
    
    # We only need the old mus for checking the change of sign
    # Maybe in the future get this out
    N,D = X.shape
    N,K = r.shape
    
    # Estimated theta
    theta = []
    
    for k in range(K):
        # We compute the weighed values given by the Momentum of the Exponential family
        rk = r[:,[k]]  # Responsabilities of all the samples for that cluster
        try:
           theta_k = distribution.theta_estimator(X, rk) # Parameters of the k-th cluster
        except RuntimeError as err:
            
#            error_type = err.args[1]
#            print err.args[0] % err.args[2]
            print """ Cluster %i degenerated during estimation""" %k           ####### HANDLE THE DEGENERATED CLUSTER #############
            theta_k = None;
        theta.append(theta_k)
    return theta

def remove_cluster(theta, pi, k):
    # This function removed the cluster k from the parameters
    theta.pop(k)
    pi = np.delete(pi, k, axis = 1)
    pi = pi / np.sum(pi)
    print "$ Cluster %i removed" % (k)
    return theta, pi

def manage_clusters(X,r, distribution, pimix, theta_new, theta_prev, deged_est_params = None, deged_params = None):
    
    """ This function will deal with the generated clusters, 
    both from the estimation and the parameters.  
    For every cluster it will check if the estimation degenerated, if it did then
    we use the handler function to set the new ones. If it is None, then they will be removed.
    
    Then we check that the pdf of the distribution can be computed, a.k.a the normalization
    constant can be computed. If it cannot be computed then we call the handler. If the result is None,
    then the cluster will be removed !! """
    
    K = len(theta_new)
    Nsam,D = X.shape
    
    clusters_change = 0  # Flag is we modify the clusters so that we dont stop
                        # due to a decrease in likelihood.
                        
    if (type(theta_prev) != type(None)):  # Not run this part for the initialization one.
        ################# Resolve the degenerated estimation clusters ################
        for k in range(K):
            if(type(theta_new[k]) == type(None)):  # Degenerated cluster during estimation
                theta_new[k] = distribution.degenerated_estimation_handler(
                        X, rk = r[:,[k]] , prev_theta_k = theta_prev[k], deged_est_params = deged_est_params )
                
                clusters_change = 1  # We changed a cluster !
                
    ################# Check if the parameters are well defined ################
    for k in range(K):
        if(type(theta_new[k]) != type(None)):  # Checking for the clusters that we are not gonna remove due to
                                           # degenerated estimation.            
#            print theta_new[k]
            if (distribution.check_degeneration_params(theta_new[k]) == 0):
                print "Cluster %i has degenerated parameters "%k
                
                if (type(r) == type(None)): # We do not have rk if this is the first initialization
                    rk = None
                else:
                    rk = r[:,[k]]
                
                # We give the current theta to this one !!! 
                theta_new[k] = distribution.degenerated_params_handler(
                        X, rk = rk , prev_theta_k = theta_new[k], deged_params = deged_params )
                
                clusters_change = 1; # We changed a cluster !
    
    ################## Last processing that you would like to do with everything ##############
    theta_new = distribution.use_chageOfClusters(theta_new, theta_prev)
    
    ############## Remove those clusters that are set to None ###################
    for k in range(K):
        k_inv = K - 1 -k
        if(type(theta_new[k_inv]) == type(None)):  # Degenerated cluster during estimation
            theta_new,pimix = remove_cluster(theta_new,pimix,k_inv)
        
        
#    kummer_check = []
#    for k in range(K):
#        try:
#            kummer_check.append(Wad.get_cp_log(D,kappas[:,k]))
#        except RuntimeError as err:
#            print "Error in Managing clusters"
#            error_type = err.args[1]
#            print err.args[0] % err.args[2]
#            print """We saturate kappa to %f. But in the next estimation the estimated kappa will also be as bad"""% (Kappa_max)
#            
#            # TODO: This could not work if the Kappa_max is still to high
#            kappas[:,k] = Kappa_max * kappas[:,k]/np.abs(kappas[:,k])
#            clusters_change = 1
#            
#    # We go backwards in case we erase, not to affect indexes
#    for k in range(K):
#        
#        if (np.abs(kappas[0,K - 1 -k]) > 1000):
#            pass
#            theta, pi = remove_cluster(theta,pi,K - 1 -k)
            
#        if (kummer_check[K - 1 -k] == 0): # If we fucked up
#            theta, pi = remove_cluster(theta,pi,K - 1 -k)

    # Maybe condition on pi as well ? If pi is too small.

#    K = theta[0].shape[1]
#    for k in range(K):
#        if (pi[0,K - 1 -k] <  0.02/K):  # Less than 5 percent of the uniform # (1/float(Nsam)
#            theta, pi = remove_cluster(theta,pi,)
#    



    return theta_new,pimix, clusters_change
    