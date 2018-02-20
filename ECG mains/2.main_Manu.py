"""
This code will:
    - Load previously generated data in 3D for both EM and HMM
        We can use the same data for EM and HMM by using the same data in EM as HMM.
    - Perform 5-fold CV of the dataset in both EM and HMM
    - Plot the evolution of LL for EM and HMM with the number of clusters
"""

# Change main directory to the main folder and import folders
import os,sys
import preprocessing as prep
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
import seaborn as sns 

import EM_lib as EMl
import pickle_lib as pkl

import Watson_estimators as Wae
import Watson_distribution as Wad

import CEM 
import HMM_libfunc as HMMlf
import general_func as gf
import CDistribution as Cdist

# sklearn elements
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import copy
plt.close("all")

folder = "./data/test_data/"
folder_HMM = "./data/HMM_data/"

## Flags
load_ERG_data = 1  # To load the data from the HMM structure
CV_EM = 1          # To perform the CV of the EM
CV_HMM = 0         # To perform the CV of the HMM
plot_combined_EM_HMM = 0 # To plot the data 
################################################################
######## Or Load the same data as for HMM 3 sets ###############
###############################################################
if (load_ERG_data):
    data,conditions, time = gf.load_real_data('data/')
    
    """ Data is a list of 3 elements, one per condition.
        Each of the elements contains an np.ndarray of (16,), corresponding to 16 people.
        This apparently also acts like numpy "list"
        Each person constains a single time series of dimensionality (451, 70) that we need to normalize
    """
    raw_data = gf.transpose_data(data)
    data_nobaseline = gf.remove_baseline_data(raw_data)
    data_norm = gf.normalize_data(data_nobaseline)
    X = gf.cut_beginning_data(data_norm)
    Ncond, Npeople = X.shape
    D = X[0,0].shape[1]

#######################################################################################
######## Create the Distribution object ##############################################
#######################################################################################

#### 1st Gaussian Distribution
Gaussian_d = Cdist.CDistribution(name = "Gaussian");
Gaussian_d.set_distribution("Gaussian")
Gaussian_d.parameters["mu_variance"]  = 1e-100
Gaussian_d.parameters["Sigma_min_init"] = 1e-13
Gaussian_d.parameters["Sigma_max_init"] = 1e-14
Gaussian_d.parameters["Sigma_min_estimation"] = 0.1
Gaussian_d.parameters["Sigma_min_distribution"] = 0.1
Gaussian_d.parameters["Sigma"] = "diagonal"

#### 2nd: Watson distribution
Watson_d = Cdist.CDistribution(name = "Watson");
Watson_d.set_distribution("Watson")
Watson_d.parameters["Num_Newton_iterations"] = 5
Watson_d.parameters["Allow_negative_kappa"] = "no"
Watson_d.parameters["Kappa_max_init"] = 2
Watson_d.parameters["Kappa_max_singularity"] =1000
Watson_d.parameters["Kappa_max_pdf"] = 1000
Watson_d.use_changeOfClusters = None

#### 3rd von MisesFisher distribution ######
vonMisesFisher_d = Cdist.CDistribution(name = "vonMisesFisher");
vonMisesFisher_d.set_distribution("vonMisesFisher")

vonMisesFisher_d.parameters["Num_Newton_iterations"] = 2
vonMisesFisher_d.parameters["Kappa_max_init"] = 20
vonMisesFisher_d.parameters["Kappa_max_singularity"] =1000
vonMisesFisher_d.parameters["Kappa_max_pdf"] = 1000

########################################################################################
#####################   Crossvalidate Using EM   ####################################
#########################################################################################
for data_rel in ['MarkovChain1','MarkovChain1']:
    ## Transform HMM lists in to a np.matrix (Nsam, Ndim) for the EM
    ### Number of clusters to crossvalidate
    Klusters =  [1,2,3,4]  # range(1,8) # 3,4,5,6,10,10,12,15
    NCVs = 1    # Number of times we do the crossvalidation
    Nfolds =  2; Npeople  # Number of folds of each crossvalidation
    
    
    ############# SET TRAINNG PARAMETERS ##################################
    Ninit = 3
    delta_ll = 0.02
    T = 30
    verbose = 1;
    clusters_relation = data_rel  # MarkovChain1  independent
    time_profiling = None
    
    ### Data structures to save the results
    logl_tr_CVs = []  # List of the training LL of all CVs   [nCVs]
    logl_val_CVs = [] # List of the validation LL of all CVs   [nCVs]

    """ Now we perform the CV of the number of clusters just for one condition """
    
    cond_i = 0
    Xdata = X[cond_i,:]
    ############################# Begin CV  ####################################
    for distro in ['watson']:#['vonMises','watson','gaussian']:
        fake_labels = np.ones(Xdata.shape[0]).flatten() # Emulation of all samples have the same class
        for nCV_i in range(NCVs):   # For every CV that we will perform
            logl_tr_CV_i = []   # List the LL values for a single CV, for all clusters and Nfolds. [NKluters] = [LL_fold1, LL_fold2,....]]
            logl_val_CV_i = []  # List the LL values for a single CV, for all clusters and Nfolds. [NKluters] = [LL_fold1, LL_fold2,....]]
            # We create the splitting of indexes
            stkfold = cross_validation.StratifiedKFold(fake_labels, n_folds = Nfolds)
            # We are going to use the same partition for all parameters to crossvalidate.
            # In this case the number of clusters
            for K in Klusters:  
                print "CrossValidating Number of Clusters %i" % K
                # Create the distributino manager with those clusters
                myDManager = Cdist.CDistributionManager()
                K_G = 0; K_W = 0; K_vMF = 0
                if distro == 'vonMises':
                    K_vMF = K
                elif distro == 'watson':
                    K_W = K
                else:
                    K_G = K

                if (K_G > 0):
                    myDManager.add_distribution(Gaussian_d, Kd_list = range(0,K_G))
                if(K_W > 0):
                    myDManager.add_distribution(Watson_d, Kd_list = range(K_G,K_W+ K_G))
                if(K_vMF > 0):
                    myDManager.add_distribution(vonMisesFisher_d, Kd_list = range(K_W+ K_G,K_W+ K_G+ K_vMF))
                    
                ll_tr_params_folds = []
                ll_val_params_folds = []
                ifold = 1
                for train_index, val_index in stkfold:  # For each partition, using the set of parameters to CV
                    print "Starting fold out of %i/%i"%(ifold,Nfolds)
                    ifold = ifold + 1
                    Xdata_train = [Xdata[indx] for indx in train_index]
                    Xdata_val = [Xdata[indx] for indx in val_index]
                
    #                print Xdata_train.shape
    #                print Xdata_val.shape
                    # Create the EM object with the hyperparameters
                    myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
                    T = T, Ninit = Ninit,  delta_ll = delta_ll, 
                    verbose = verbose, time_profiling = time_profiling)
                    ## Perform EM with the hyperparameters already set, we just fit the data and position init
                    theta_init = None; model_theta_init = None
                    ############# PERFORM THE EM #############
                    logl,theta_list,mode_theta_list = myEM.fit(Xdata_train, model_theta_init = model_theta_init, theta_init = theta_init) 
            
                    ## Compute the likelihoods for train and test divided by the number of samples !! 
                    ## We are computing the normalized Likelihood !! Likelihood per sample !!
                    new_ll = myEM.get_loglikelihood(Xdata_train,myDManager, theta_list[-1],mode_theta_list[-1])/train_index.size
                    ll_tr_params_folds.append(copy.deepcopy(new_ll))
                    new_ll = myEM.get_loglikelihood(Xdata_val,myDManager, theta_list[-1],mode_theta_list[-1])/val_index.size
                    ll_val_params_folds.append(copy.deepcopy(new_ll))
                
                logl_tr_CV_i.append(copy.deepcopy(ll_tr_params_folds))
                logl_val_CV_i.append(copy.deepcopy(ll_val_params_folds))
        
            logl_tr_CVs.append(logl_tr_CV_i)
            logl_val_CVs.append(logl_val_CV_i)

        # Create alterego variables for using in emergency later
        logl_tr_CVs_EM_save_for_later = copy.deepcopy(logl_tr_CVs)
        logl_val_CVs_EM_save_for_later = copy.deepcopy(logl_val_CVs)
        ################################################################################################################
        ################################### PLOTTING THE RESULTS #####################################################
        ################################################################################################################
        ### Reschedule the data. We are going to for a single CV:
        # Compute the mean LL of the CVs for train and validation
        
        for i in range(len(logl_tr_CVs)):
            data_tr = np.array(logl_tr_CVs[i])
            data_val = np.array(logl_val_CVs[i])
            plt.close()
            ax = sns.tsplot(data_tr.T, time=Klusters, ci=[68, 95], color='blue', condition='Train data')
            ax = sns.tsplot(data_val.T, time=Klusters, ci=[68, 95], color='red', condition='Validation data')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('loglikelihood')
            plt.savefig('output/CV_'+data_rel+'_'+distro+str(i)+'.png')


