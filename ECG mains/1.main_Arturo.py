"""
This code will load previously generated data in 3D. 
It can load EM data or HMM.
It will plot the data if specified (notice colored clusters only available in EM)
It will perform the EM.
"""

# Change main directory to the main folder and import folders
import os,sys
#sys.path.append('../')
#base_path = os.path.abspath('')
sys.path.append(os.path.abspath('../'))
os.chdir("../")
import import_folders

# Official libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Own libraries
from graph_lib import gl
import sampler_lib as sl
import EM_libfunc as EMlf
import EM_lib as EMl
import copy
import pickle_lib as pkl

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf

import CEM as CEM 
import CDistribution as Cdist
plt.close("all")

folder = "./data/test_data/"
folder_HMM = "./data/HMM_data/"

#TODO:
#   - CV for Watson
#   - CV for vonMises
#   - predictions LOO   

def main ():
    data,conditions = load_real_data('data/')
    X =  normalize_data(data)
    Xdata = X[1,1]
    D = Xdata.shape[1]
    #######################################################################################################################
    ######## Create the Distribution object ####################
    Watson_d = Cdist.CDistribution(name = "Watson");
    
    Watson_d.pdf_log_K = Wad.Watson_K_pdf_log
    Watson_d.init_params = Wad.init_params
    Watson_d.theta_estimator = Wae.get_Watson_muKappa_ML 
    
    ## For degeneration
    Watson_d.degenerated_estimation_handler = Wad.degenerated_estimation_handler 
    Watson_d.degenerated_params_handler = Wad.degenerated_params_handler 
    Watson_d.check_degeneration_params = Wad.check_params
    
    ## For optimization
    Watson_d.get_Cs_log = Wad.get_Cs_log
    
    ## Optional for more complex processing
    Watson_d.use_chageOfClusters = Wad.avoid_change_sign_centroids
    #######################################################################################################################
    K_list = [6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,25]
    df_list = CV_LOO(X,K_list,conditions,Watson_d)

def CV_LOO(X_list,K_list,conditions,Watson_d):
    num_conditions,num_subjects = X_list.shape
    df_list = []
    for i in range(num_conditions):
        condition = X_list[i,:]
        df = pd.DataFrame(index = K_list, columns=['train','LOO'])
        for k in K_list:
            test_ll = []; train_ll = []
            for j in range(num_subjects):
                print"====================================================================="
                print i,k,j
                print "============================================================="

                LOO, rest = condition[j], np.delete(condition,j)
                rest_np = rest[0]
                for x in rest[1:]:
                    rest_np = np.concatenate([rest_np,x])
                ############### SET HYPERPARAMETERS DISTRIBUTION ######################
                kappa_max_init = 50;
                kappa_max_estimation = 1000;
                kappa_max_distribution = 1000;

                ############# SET TRAINNG PARAMETERS ##################################
                K = k
                Ninit = 5
                delta_ll = 0.02
                T = 10
                verbose = 0;
                #######################################################################
                # Create the EM object and fit the data to it.
                myEM = CEM.CEM( distribution = Watson_d, init_hyperparams = [kappa_max_init],
                               deged_est_params = [kappa_max_estimation], deged_params = [kappa_max_distribution],
                                K = K, Ninit = Ninit, delta_ll = delta_ll, T = T, verbose = verbose)
                
                new_train_ll,theta_list,pimix_list = myEM.fit(rest_np, pi_init = None,theta_init = None) 
                r_log, new_test_ll = EMlf.get_r_and_ll(LOO,Watson_d,theta_list[-1],pimix_list[-1])
                test_ll.append(new_test_ll)
                train_ll.append(new_train_ll[-1]/(num_subjects-1))
            df.set_value(k,'LOO',np.array(test_ll).mean())
            df.set_value(k,'train',np.array(train_ll).mean())
            df.set_value(k,'LOO_std',np.array(test_ll).std())
            df.set_value(k,'train_std',np.array(train_ll).std())
            df.to_csv('./wat_'+conditions[i]+'.csv',sep=',')
        df_list.append(df)
    return df_list                

def load_real_data(path = 'data/',file_name='face_scrambling_ERP.mat'):
    import scipy.io as sio
    mat = sio.loadmat(path+file_name)
    data = mat['X_erp']
    conditions = [x[0] for x in  mat['CONDITIONS'][0]]
    return data,conditions

def load_fake_data(path = 'data/'):
    from os import listdir
    csv = [x for x in listdir(path) if x.find('.csv')!=-1]
    data = [np.array(pd.read_csv(path+X,sep=','))  for X in csv]
    X = np.concatenate(data,axis=0)
    return X

def normalize_data(data):
    for i,row in enumerate(data):
        for j,x in enumerate(row):
            data[i,j] = normalize_subject(x)
    return data

def normalize_subject(data):
    d,N = data.shape
    for i in range(N):
        data[:,i] = data[:,i]/np.linalg.norm(data[:,i])
    return data.T

if __name__ == "__main__":
    main()
