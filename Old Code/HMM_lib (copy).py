import distributions_func as difu
import numpy as np
import HMM_libfunc2 as HMMlf
import copy

def HMM(I,data,delta,R):
    # Input
    # S = Number of States
    # data = Data
    # alpha = minimu step for convergence (If negative it does not check it)
    # R = Max iterations
    # 
    # Output
    # pi = pi parameters
    # A =  A parameters
    # B =  B parameters
    # logL = complete log-likelihood of each iteration
    #
    #  [pi,A,B,alfa] = HMM(5,observed,0.1,1)


    #########################################################
    # data = list of realizations, every realization has N samples !! 
    # It can vary from realization to realization

    N = len(data)       # Number of Realizations of the HMM
    D = data[0].shape[1]; # Dimension of multidimensial bernoulli
    T = data[0].shape[0]; # Number of samples of the HMM

    #********************************************************
    #*********** INITIALIZATION *****************************
    #********************************************************

    # Here we will initialize the  parameters of the HMM, that is,the initial
    # probabilities of the state "pi", the transition probabilities "A" and the
    # parameters of the probability functions "B"

    # Initial probabilities
    # We set the Initial probabilities with uniform discrete distribution, this
    # way, the a priori probability of any vector to belong to any component is
    # the same.
    
    pi = np.ones((1,I));
    pi = pi*(1/float(I));

    # Transition probabilities "A"
    # We set the Transition probabilities with uniform discrete distribution, this
    # way, the a priori probability of going from a state i to a state j is
    # the same, no matter the j.

    A = np.ones((I,I));   #A(i,j) = aij = P(st = j | st-1 = i)  sum(A(i,:)) = 1
    for i in range(I):
        A[i,:] =  A[i,:]*(1/float(I));

    # Parameters of the probability functions "B"
    # Give random values to the transit parameters. Since in this case, all
    # theta parameters theta(d,k), are the Expected value of a Bernoulli, we
    # asign values to them at random accoding to a uniform continuous
    # distribution in the support (0,1).
    
    mus = np.random.randn(D,I);
    mus = difu.normalize_data(mus.T).T
    kappas = np.ones((1,I)) * 20
    
    # We store the parameter of the clusters in B
    B = [mus, kappas]
    
    #Initialize log-likelihood to 0
    
    #*********************************************************
    #*********** ITERATIONS OF THE HMM ************************
    #*********************************************************
    logl = []   # List where we store the likelihoos
    mus_list = [] # List where we store the mus
    kappas_list = [] # Kappas
    pi_list = []
    A_list = []
    
    mus_list.append(copy.deepcopy(mus))
    kappas_list.append(copy.deepcopy(kappas))
    pi_list.append(copy.deepcopy(pi))
    A_list.append(copy.deepcopy(A))
    
    for r in range(R):         # For every iteration of the EM algorithm
        print "Iteration %i"%(r)
        #******************************************************  
        #*********** E Step ***********************************
        #******************************************************
        
#        print "pi paramters"
#        print pi
#        print "mus"
#        print B[0].T
#        print "kappas"
#        print B[1]
#        print "A"
#        print A
        
        # In this step we calculate the alfas, betas, gammas and fis matrices
        
        # Compute the initial incomplete-loglikelihood
        
        if (r == 0):
            ## ALPHA is recomputed ar the end to 
            alpha = HMMlf.get_alfa_matrix_log(A,B,pi,data);
            ll = HMMlf.get_HMM_Incomloglike(A,B,pi,data,alpha)
            print "Initial loglikelihood: %f" % ll
        #***************************************************** 
        #****** Convergence Checking *************************
        #*****************************************************
        
        # Probabilities get vanishingly small as T -> 0
        # Maybe take logarithms ?
        
#        print alpha[0,0,:]  # I * N * T
#        print alpha[0,1,:]
        
        beta = HMMlf.get_beta_matrix_log( A,B,data);
        # Probabilities get vanishingly small as t -> 0
        # Maybe take logarithms ?
#        print beta.shape
#        print alpha[0,0,:]  # I * N * T
#        print beta[0,1,:]
       
        gamma = HMMlf.get_gamma_matrix_log(alpha,beta );
        
#        print gamma [0,0,:]
        fi = HMMlf.get_fi_matrix_log(A,B, alpha,beta,data );
        
#        print fi [0,0,:]
        
        #*****************************************************   
        #*********** M Step ***********************************
        #*****************************************************
        # In this step we calculate the next parameters of the HMM
        
        gamma = np.exp(gamma)
        fi = np.exp(fi)
        
        # Calculate new initial probabilities
        N_gamma = []
        for n in range(N):
            N_gamma.append = np.sum(gamma[n][:,0]);
        N_gamma = np.sum(N_gamma)
        
        for i in range(I):
            aux = []
            for n in range(N):
                aux.append(gamma[n][i,:])

            N_i_gamma = np.sum(aux)
            pi[0,i] = N_i_gamma/N_gamma;
            
        pi = pi.reshape(1,pi.size)
#        print "pi"
#        print pi
        
        # Calculate transition probabilities A
        for i in range(I):
            E_i_fi = []
            # Calculate vector ai = [ai1 ai2 ... aiJ]  sum(ai) = 1
            for n in range(N): 
                E_i_fi.append(np.sum(np.sum(fi[n][i,:,:])))
            E_i_fi = np.sum(E_i_fi)
            
            E_ij_fi = []
            for j in range(I):
                for n in range(N): 
                    E_ij_fi.append(np.sum(fi[n][i,j,:]))
                E_ij_fi = np.sum(E_ij_fi)
                A[i,j] = E_ij_fi/E_i_fi;
                
#        print "A"
#        print A
        # Calculate the paramters B
    
        for i in range(I): # For every cluster
             # We compute the gamma normalization of the cluster
            aux = []
            for n in range(N):
                aux.append(gamma[n][i,:])

            N_i_gamma = np.sum(aux)
            # The contribution of every sample is weighted by gamma[i,n,t];
            # The total responsibility of the cluster for the samples is N_i_gamma

            rk = gamma[i,:,:].flatten()
            rk = rk.reshape(rk.size,1)
            
            X = data[0]
            for ix in range(1,N):
                X = np.concatenate((X, data[ix]), axis = 0)
                
            new_mu = difu.get_Weighted_MLMean(rk,X)  
            
            signs = np.sum(np.sign(new_mu *  mus[:,i]))
#            print signs
            if (signs < 0):
                mus[:,i] = -new_mu
            else:
                mus[:,i] = new_mu
                
            kappas[:,i] = difu.get_Weighted_MLkappa(rk,mus[:,i],X)
        
            B = [mus, kappas]
        #********************************************************* 
        #****** Calculate Incomplete log-likelihood  *************
        #*********************************************************
    
        # Remember that the Incomplete log-likelihood could decrease with
        # the number of iterations at some stage since the EM algorith 
        # maximizes the Complete log-likelihood (They are different)
    
        # Calculate Incomplete log-likelihood with the Forward Algorithm
        alpha = HMMlf.get_alfa_matrix_log(I,N,T, A,B,pi,data);
        new_ll = 0;
        for n in range(N):    # For every HMM sequence
            new_ll = new_ll + HMMlf.sum_logs(alpha[:,n,T-1]);

        logl.append(new_ll);
        
        print "Loglkelihood: %f " % new_ll
        #***************************************************** 
        #****** Convergence Checking *************************
        #*****************************************************
 
        logl.append(new_ll)
        mus_list.append(copy.deepcopy(mus))
        kappas_list.append(copy.deepcopy(kappas))
        pi_list.append(copy.deepcopy(pi))
        A_list.append(copy.deepcopy(A))
    
        if(np.abs(new_ll-ll) <= delta):
            break;
        else:
            ll = new_ll;

#        print'  R:'
#        print r;

    return logl,mus_list,kappas_list,pi_list, A_list


