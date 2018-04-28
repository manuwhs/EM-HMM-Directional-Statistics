"""
This code will:
    - Load previously generated data in 3D for both EM and HMM
        We can use the same data for EM and HMM by using the same data in EM as HMM.
    - Perform 5-fold CV of the dataset in both EM and HMM
    - Plot the evolution of LL for EM and HMM with the number of clusters
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
from scipy.stats import multivariate_normal
import EM_lib as EMl
import pickle_lib as pkl
import Gaussian_distribution as Gad
import Gaussian_estimators as Gae
import Watson_estimators as Wae
import Watson_distribution as Wad

import CEM
import HMM_libfunc as HMMlf
import general_func as gf
import CDistribution as Cdist
import basicMathlib as bMA
# sklearn elements
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import copy
plt.close("all")

folder = "./data/test_data/"
folder_HMM = "./data/HMM_data/"
folder_images = "../pics/Trapying/EM_HMM/"

## Flags
CV_EM = 1          # To perform the CV of the EM
CV_HMM = 0         # To perform the CV of the HMM
generate_Gaussian_data = 1 # To plot the data 


######################################################################################
######## Create the Distribution object ##############################################
#######################################################################################
Gaussian_d = Cdist.CDistribution(name = "Gaussian");
Gaussian_d.set_distribution("Gaussian")
Gaussian_d.parameters["Sigma"] = "diagonal"

if (generate_Gaussian_data):
    N1 = 200
    N2 = 400
    N3 = 600
    
    mu1 = np.array([[2],[4]])
    cov1 = np.array([[1,0.4],
                     [0.4,1]])

    mu2 = np.array([[-2],[-2]])
    cov2 = np.array([[1.5,-0.2],
                     [-0.2,1.5]])
    
    mu3 = np.array([[3],[-4]])
    cov3 = np.array([[2,-0.8],
                     [-0.8,2]])
    
    X1 = np.random.multivariate_normal(mu1.flatten(), cov1, N1).T
    X2 = np.random.multivariate_normal(mu2.flatten(), cov2, N2).T
    X3 = np.random.multivariate_normal(mu3.flatten(), cov3, N3).T
    
    ######## Plotting #####
    gl.init_figure();
    ## First cluster
    ax1 = gl.scatter(X1[0,:],X1[1,:], labels = ["Gaussian Generated Data", "x1","x2"], 
                     legend = ["K = 1"], color = "r",alpha = 0.5)
    mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu1, Sigma = cov1, Chi2val = 2.4477)
    r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
    gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2
             ,AxesStyle = "Normal2", color = "r")
    
    ## Second cluster
    ax1 = gl.scatter(X2[0,:],X2[1,:], legend = ["K = 2"], color = "b", alpha = 0.5)
    mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu2, Sigma = cov2, Chi2val = 2.4477)
    r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
    gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2,AxesStyle = "Normal2", color = "b")
    
    ## Third cluster
    ax1 = gl.scatter(X3[0,:],X3[1,:], legend = ["K = 3"], color = "g",alpha = 0.5)
    mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu3, Sigma = cov3, Chi2val = 2.4477)
    r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
    gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--", lw = 2,AxesStyle = "Normal2", color = "g")
    
    
    ############ ESTIMATE THEM ################
    theta1 = Gae.get_Gaussian_muSigma_ML(X1.T)
    print ("mu1:")
    print (theta1[0])
    print ("Sigma1")
    print(theta1[1])
    
    ############## Estimate Likelihood ###################
    ll = Gad.Gaussian_pdf_log (X1, [mu1,cov1])
    ll2 = []
    for i in range (ll.size):
        ll2.append( multivariate_normal.logpdf(X1[:,i], mean=mu1.flatten(), cov=cov1))
    ll2 = np.array(ll2).reshape(ll.shape)
    
    print ("ll ours")
    print (ll.T)
    print ("ll scipy")
    print (ll2.T)
    print ("Difference in ll")
    print ((ll - ll2).T)
    
    ###### Multiple clusters case
    ll_K = Gad.Gaussian_K_pdf_log(X1, [[mu1,cov1],[mu2,cov2]])
    

########################################################################################
#####################   Crossvalidate Using EM   ####################################
#########################################################################################
if (CV_EM):
    ## Transform HMM lists in to a np.matrix (Nsam, Ndim) for the EM
    Xdata = np.concatenate((X1,X2,X3), axis =1).T
    N,D = Xdata.shape
    
    ### Number of clusters to crossvalidate
    Klusters = [1,2,3,4,5] # range(1,8) # 3,4,5,6,10,10,12,15
    NCVs = 1    # Number of times we do the crossvalidation
    Nfolds = 2  # Number of folds of each crossvalidation
    
    #######################################################################################
    ######## Create the Distribution object ##############################################
    #######################################################################################
    #### 1st Gaussian Distribution
    Gaussian_d = Cdist.CDistribution(name = "Gaussian");
    Gaussian_d.set_distribution("Gaussian")
    #### 2nd: Watson distribution
    Watson_d = Cdist.CDistribution(name = "Watson");
    Watson_d.set_distribution("Watson")

    ############# SET TRAINNG PARAMETERS ##################################
    Ninit = 5
    delta_ll = 0.02
    T = 20
    verbose = 1;
    clusters_relation = "MarkovChain1"   # MarkovChain1  independent
    time_profiling = None
    
    ### Data structures to save the results
    logl_tr_CVs = []  # List of the training LL of all CVs   [nCVs]
    logl_val_CVs = [] # List of the validation LL of all CVs   [nCVs]
    
    ############################# Begin CV  ####################################
    fake_labels = np.ones(Xdata.shape[0]).flatten() # Emulation of all samples have the same class
    for nCV_i in range(NCVs):   # For every CV that we will perform
        logl_tr_CV_i = []   # List the LL values for a single CV, for all clusters and Nfolds. [NKluters] = [LL_fold1, LL_fold2,....]]
        logl_val_CV_i = []  # List the LL values for a single CV, for all clusters and Nfolds. [NKluters] = [LL_fold1, LL_fold2,....]]
        # We create the splitting of indexes
        stkfold = cross_validation.StratifiedKFold(fake_labels, n_folds = Nfolds, shuffle= True)
        # We are going to use the same partition for all parameters to crossvalidate.
        # In this case the number of clusters
        for K in Klusters:  
            
            # Create the distributino manager with those clusters
            myDManager = Cdist.CDistributionManager()
            K_G = K       # Number of clusters for the Gaussian Distribution
            K_W = 0
            if (K_G > 0):
                myDManager.add_distribution(Gaussian_d, Kd_list = range(0,K_G))
            if(K_W > 0):
                myDManager.add_distribution(Watson_d, Kd_list = range(K_G,K_W+ K_G))
                
            print "CrossValidating Number of Clusters %i" % K
            ll_tr_params_folds = []
            ll_val_params_folds = []
            ifold = 1
            for train_index, val_index in stkfold:  # For each partition, using the set of parameters to CV
                print "Starting fold out of %i/%i"%(ifold,Nfolds)
                ifold = ifold + 1
                Xdata_train = Xdata[train_index,:]
                Xdata_val = Xdata[val_index,:]
            
                # Create the EM object with the hyperparameters
                # Create the EM object with the hyperparameters
                myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
                T = T, Ninit = Ninit,  delta_ll = delta_ll, 
                verbose = verbose, time_profiling = time_profiling)
                ## Perform EM with the hyperparameters already set, we just fit the data and position init
                theta_init = None; model_theta_init = None
                ############# PERFORM THE EM #############
                logl,theta_list,mode_theta_list = myEM.fit([Xdata], model_theta_init = model_theta_init, theta_init = theta_init) 
                
                ## Compute the likelihoods for train and test divided by the number of samples !! 
                ## We are computing the normalized Likelihood !! Likelihood per sample !!
                new_ll = myEM.get_loglikelihood([Xdata_train],myDManager, theta_list[-1],mode_theta_list[-1])/train_index.size
                ll_tr_params_folds.append(copy.deepcopy(new_ll))
                new_ll = myEM.get_loglikelihood([Xdata_val],myDManager, theta_list[-1],mode_theta_list[-1])/val_index.size
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

    logl_tr_CVs = logl_tr_CVs_EM_save_for_later
    logl_val_CVs = logl_val_CVs_EM_save_for_later
    
    for i in range(len(logl_tr_CVs)):
        mean_tr_ll =  []
        mean_val_ll = []
        std_tr_ll = []
        std_val_ll = []
        for k_i in range(len(Klusters)):
            mean_tr_ll.append(np.mean(logl_tr_CVs[i][k_i]))
            mean_val_ll.append(np.mean(logl_val_CVs[i][k_i]))
            std_tr_ll.append(np.std(logl_tr_CVs[i][k_i]))
            std_val_ll.append(np.std(logl_val_CVs[i][k_i]))
            # For each CVs
    
        mean_tr_ll = np.array(mean_tr_ll)
        mean_val_ll= np.array(mean_val_ll)
        std_tr_ll= np.array(std_tr_ll)
        std_val_ll= np.array(std_val_ll)
            
gl.init_figure()
gl.plot(Klusters,mean_tr_ll, legend = ["Mean Train LL (EM)"], 
        labels = ["Validation of Number of clusters for a %i-CV EM."%Nfolds,"Number of clusters (K)","Average LL of a sample"], 
        lw = 3, color = "k")

gl.plot(Klusters,mean_tr_ll + 2*std_tr_ll , color = "k", nf = 0, lw = 1, ls = "--", legend = ["Mean Train LL +- 2std"])
gl.plot(Klusters,mean_tr_ll - 2*std_tr_ll , color = "k", nf = 0, lw = 1,ls = "--")
gl.fill_between(Klusters, mean_tr_ll - 2*std_tr_ll, mean_tr_ll + 2*std_tr_ll, c = "k", alpha = 0.5)
for i in range(len(logl_tr_CVs)):
    for k_i in range(len(Klusters)):
        gl.scatter(np.ones((len(logl_tr_CVs[i][k_i]),1))*Klusters[k_i], logl_tr_CVs[i][k_i], color = "k", alpha = 0.2, lw = 1)
    
gl.plot(Klusters,mean_val_ll, nf = 0, color = "r",
        legend = ["Mean Validation LL (EM)"], lw = 3)
gl.plot(Klusters,mean_val_ll + 2*std_val_ll , color = "r", nf = 0, lw = 1, ls = "--", legend = ["Mean Validation LL +- 2std"])
gl.plot(Klusters,mean_val_ll - 2*std_val_ll , color = "r", nf = 0, lw = 1, ls = "--")
gl.fill_between(Klusters, mean_val_ll - 2*std_val_ll, mean_val_ll + 2*std_val_ll, c = "r", alpha = 0.1)

for i in range(len(logl_tr_CVs)):
    for k_i in range(len(Klusters)):
        gl.scatter(np.ones((len(logl_val_CVs[i][k_i]),1))*Klusters[k_i], logl_val_CVs[i][k_i], color = "r", alpha = 0.5, lw = 1)

gl.savefig(folder_images + 'EM_Gaussian_CV_artificial_data.png', 
       dpi = 100, sizeInches = [12, 6])
