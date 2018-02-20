
import distributions_func as difu
import numpy as np

def sum_logs(log_vector):
    # This functions sums a vector of logarithms
    # alfa[i,n,t] = np.logaddexp(aux, alfa[i,n,t])

    log_vector = np.array(log_vector).flatten()
    a0 = log_vector[0]
    
    others = np.array(log_vector[1:])
    
    caca = np.sum(np.exp(others - a0))
    
    result = a0 + np.log(1 + caca)
    return result

def get_alfa_matrix( I,N,T, A,B,pi,data ):
    alfa = np.zeros((I,N,T));
    # Calculate  alfa(1,:,:)
    
    for n in range(N): # For every chain
        for i in range(I):  # For every state
            alfa[i,n,0] = pi[:,i] * difu.Watson_pdf(data[n][0,:], B[0][:,i], B[1][:,i]);

    # Calculate the rest of the alfas recursively
    for t in range(1, T):           # For every time instant
        for n in range(N):          # For every chain
            for i in range(I):      # For every state
                for j in range(I):  # For every state
                    alfa[i,n,t] = alfa[i,n,t] + A[j,i]*alfa[j,n,t-1];
                alfa[i,n,t] =  difu.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i])* alfa[i,n,t];
                         
    return alfa

def get_alfa_matrix_log( I,N,T, A,B,pi,data ):
    alfa = np.zeros((I,N,T));
    # Calculate  alfa(1,:,:)
    
    for n in range(N): # For every chain
        for i in range(I):  # For every state
            alfa[i,n,0] = np.log( pi[:,i] *difu.Watson_pdf(data[n][0,:], B[0][:,i], B[1][:,i]));

    # Calculate the rest of the alfas recursively
    for t in range(1, T):           # For every time instant
        for n in range(N):          # For every chain
            for i in range(I):      # For every state
                aux_vec = []
                for j in range(I):  # For every state
                    alfa[i,n,t] = alfa[i,n,t] + A[j,i]*alfa[j,n,t-1];
                    aux_vec.append(np.log(A[j,i]) + alfa[j,n,t-1])

                alfa[i,n,t] = sum_logs(aux_vec)
                alfa[i,n,t] =  np.log(difu.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i])) + alfa[i,n,t];
                
#                print np.log(difu.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i]))
#                print    np.log(difu.Watson_pdf(data[n][t,:], B[0][:,i], B[1][:,i]))# alfa[i,n,t] 
    return alfa
    
def  get_beta_matrix( I,N,T, A,B,data ):
    beta = np.zeros((I,N,T));
    # Calculate  beta(1,:,:)
    
    for n in range(N):
        for i in range(I):
            beta[i,n,T-1] = 1;

    # Calculate the rest of the betas recursively
    for t in range(T-2,-1,-1):  # For every time instant backwards
        for n in range(N):     # For every chain
            for i in range(I):  # For every state
                for j in range(I):  # For every state
                    beta[i,n,t] = beta[i,n,t]+ A[i,j] * beta[j,n,t+1] * \
                        difu.Watson_pdf(data[n][t,:], B[0][:,j], B[1][:,j]);

    return beta

def  get_beta_matrix_log( I,N,T, A,B,data ):
    beta = np.zeros((I,N,T));
    # Calculate  beta(1,:,:)
    
    for n in range(N):
        for i in range(I):
            beta[i,n,T-1] = np.log(1);

    # Calculate the rest of the betas recursively
    for t in range(T-2,-1,-1):  # For every time instant backwards
        for n in range(N):     # For every chain
            for i in range(I):  # For every state
                aux_vec = []
                for j in range(I):  # For every state
                    aux = np.log(A[i,j]) +  beta[j,n,t+1] + np.log(difu.Watson_pdf(data[n][t,:], B[0][:,j], B[1][:,j]))
                    aux_vec.append(aux)
                beta[i,n,t] = sum_logs(aux_vec)

    return beta


def  get_fi_matrix( I,N,T,A,B, alpha,beta,data ):
    fi = np.zeros((I,I,N,T-1));
    for t in range (0, T-1):
        for n in range(N):
            for i in range(I):
                for j in range(I):
                    fi[i,j,n,t] = alpha[i,n,t] * A[i,j] * beta[j,n,t+1] * \
                    difu.Watson_pdf(data[n][t+1,:], B[0][:,j], B[1][:,j])

    for t in range (0, T-1):
        for n in range(N):
            # Normalize to get the actual fi
            fi[:,:,n,t] = fi[:,:,n,t]/np.sum(np.sum(fi[:,:,n,t]));  

    return fi

def  get_fi_matrix_log( I,N,T,A,B, alpha,beta,data ):
    fi = np.zeros((I,I,N,T-1));
    for t in range (0, T-1):
        for n in range(N):
            for i in range(I):
                for j in range(I):
                    fi[i,j,n,t] = alpha[i,n,t] + np.log(A[i,j]) + beta[j,n,t+1] + \
                     np.log(difu.Watson_pdf(data[n][t+1,:], B[0][:,j], B[1][:,j]))

    for t in range (0, T-1):
        for n in range(N):
            # Normalize to get the actual fi
            fi[:,:,n,t] = fi[:,:,n,t] - sum_logs(fi[:,:,n,t]);  

    return fi
    

def get_gamma_matrix( I,N,T, alpha,beta ):
    gamma = np.zeros((I,N,T));
    for t in range(T):
        for n in range(N):
            for i in range(I):
                gamma[i,n,t] = alpha[i,n,t] * beta[i,n,t];

    for t in range(T):
        for n in range(N):
            #Normalize to get the actual gamma
            gamma[:,n,t] = gamma[:,n,t] / np.sum(gamma[:,n,t]);  

    return gamma


def get_gamma_matrix_log( I,N,T, alpha,beta ):
    gamma = np.zeros((I,N,T));
    for t in range(T):
        for n in range(N):
            for i in range(I):
                gamma[i,n,t] = alpha[i,n,t] + beta[i,n,t];

    for t in range(T):
        for n in range(N):
            #Normalize to get the actual gamma
            gamma[:,n,t] = gamma[:,n,t] - sum_logs(gamma[:,n,t]);  

    return gamma
