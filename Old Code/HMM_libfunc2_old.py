
import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf

import numpy as np

def sum_logs(log_vector):
    # This functions sums a vector of logarithms
    # alfa[i,n,t] = np.logaddexp(aux, alfa[i,n,t])

    log_vector = np.array(log_vector).flatten()
    log_vector = np.sort(log_vector) # Sorted min to max
    a0 = log_vector[-1]
    
    others = np.array(log_vector[:-1])
    
    caca = np.sum(np.exp(others - a0))
    
    result = a0 + np.log(1 + caca)
    return result
  
def get_alfa_matrix_log( A,B,pi,data ):
    I = A.shape[0]
    N = len(data)
    D = data[0].shape[1]
    T = [len(x) for x in data]
    
    kappas = B[1]
    cp_logs = []
    
    for i in range(I):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,i]))
        
    alfa = [];
    
    # Calculate first sample
    for n in range(N): # For every chain
        alfa.append(np.zeros((I,T[n])));
        for i in range(I):  # For every state
            alfa[n][i,0] = np.log( pi[:,i]) + Wad.Watson_pdf_log(data[n][0,:], B[0][:,i], B[1][:,i], cp_log = cp_logs[i]);

    # Calculate the rest of the alfas recursively
    for n in range(N):          # For every chain
        for t in range(1, T[n]):           # For every time instant
            for i in range(I):      # For every state
                aux_vec = []
                for j in range(I):  # For every state
                    aux_vec.append(np.log(A[j,i]) + alfa[n][j,t-1])

                alfa[n][i,t] = sum_logs(aux_vec)
                alfa[n][i,t] =  Wad.Watson_pdf_log(data[n][t,:], B[0][:,i], B[1][:,i], cp_log = cp_logs[i]) + alfa[n][i,t] ;
                
#                print np.log(difu.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i]))
#                print    np.log(difu.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i]))# alfa[i,n,t] 
    return alfa
    
def  get_beta_matrix_log( A,B,data ):
    I = A.shape[0]
    N = len(data)
    D = data[0].shape[1]
    T = [len(x) for x in data]
    
    kappas = B[1]
    cp_logs = []
    
    for i in range(I):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,i]))
    
    beta = [];
    
    # Calculate the last sample
    for n in range(N): # For every chain
        beta.append(np.zeros((I,T[n])));
        for i in range(I):
            beta[n][i,-1] = np.log(1);

    # Calculate the rest of the betas recursively
    for n in range(N):     # For every chain
        for t in range(T[n]-2,-1,-1):  # For every time instant backwards
            for i in range(I):  # For every state
                aux_vec = []
                for j in range(I):  # For every state
                    aux = np.log(A[i,j]) +  beta[n][j,t+1] + Wad.Watson_pdf_log(data[n][t,:], B[0][:,j], B[1][:,j],cp_log = cp_logs[j])
                    aux_vec.append(aux)
                beta[n][i,t] = sum_logs(aux_vec)

    return beta
    
def  get_fi_matrix_log( A,B, alpha,beta,data ):
    I = A.shape[0]
    N = len(data)
    D = data[0].shape[1]
    T = [len(x) for x in data]
    
    kappas = B[1]
    cp_logs = []
    
    for i in range(I):
        cp_logs.append(Wad.get_cp_log(D,kappas[:,i]))

    fi = []
    
    for n in range(N):
        fi.append(np.zeros((I,I,T[n]-1)))
        for t in range (0, T[n]-1):
            for i in range(I):
                for j in range(I):
                    fi[n][i,j,t] = alpha[n][i,t] + np.log(A[i,j]) + beta[n][j,t+1] + \
                     Wad.Watson_pdf_log(data[n][t+1,:], B[0][:,j], B[1][:,j], cp_log = cp_logs[j])

    for n in range(N):
        for t in range (0, T[n]-1):
            # Normalize to get the actual fi
            fi[n][:,:,t] = fi[n][:,:,t] - sum_logs(fi[n][:,:,t]);  

    return fi
    
def get_gamma_matrix_log( alpha,beta ):
    I = alpha[0].shape[0]
    N = len(alpha)
    T = [x.shape[1] for x in alpha]

    gamma = []
    
    for n in range(N):
        gamma.append(np.zeros((I,T[n])))
        for t in range (0, T[n]):
            for i in range(I):
                gamma[n][i,t] = alpha[n][i,t] + beta[n][i,t];
    
    for n in range(N):
        for t in range(T[n]):
            #Normalize to get the actual gamma
            gamma[n][:,t] = gamma[n][:,t] - sum_logs(gamma[n][:,t]);  

    return gamma

def get_HMM_Incomloglike(A,B,pi,data, alpha = []):

    N = len(data)
    # Check if we have been given alpha so we do not compute it
    if (len(alpha) == 0):
        alpha = get_alfa_matrix_log(A,B,pi,data)
    new_ll = 0
    for n in range(N):    # For every HMM sequence
        new_ll = new_ll + sum_logs(alpha[n][:,-1]);
        
    return new_ll

def get_errorRate(real, pred):
    Nfails = 0
    Ntotal = 0
    for i in range(len(real)):
        T = np.array(real[i]).size
        
        for t in range(T):
    #        print decoded[i][t]
    #        print  HMM_list[i][t]
            if (int(real[i][t]) != int( pred[i][t])):
                Nfails += 1
            Ntotal += 1
    Failure = 100 * float(Nfails)/Ntotal
    return Failure

def match_clusters(mus_1, kappas_1, mus_2, kappas_2):
    # The index of the new clusters do not have to match with the order
    # of our previous cluster so we will assign to each new cluster the index
    # that is most similar to the ones we had 
    
    # Initially we will chose the one with lower distance between centroids
    
    pass

