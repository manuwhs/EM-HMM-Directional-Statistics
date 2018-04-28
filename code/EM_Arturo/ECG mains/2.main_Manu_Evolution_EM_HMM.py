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

import EM_lib as EMl
import pickle_lib as pkl

import Watson_estimators as Wae
import Watson_distribution as Wad

import CEM
import HMM_libfunc as HMMlf
import general_func as gf
import CDistribution as Cdist
import utilities_lib as ul
# sklearn elements
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import copy
plt.close("all")

folder = "./data/test_data/"
folder_HMM = "./data/HMM_data/"
folder_images = "../pics/Trapying/EM_HMM/ECG/"
ul.create_folder_if_needed(folder_images)

## Flags
load_ERG_data = 1  # To load the data from the HMM structure
EM_evolution = 1         # To perform the CV of the EM
HMM_evolution = 1         # To perform the CV of the HMM
plot_combined_EM_HMM = 1 # To plot the data 
plot_time_stuff = 1
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
    time = time.T[50:,0]
    raw_data = gf.transpose_data(data)
    data_nobaseline = gf.remove_baseline_data(raw_data)
    data_nobaseline = gf.normalize_data(data_nobaseline)
    
    X = gf.cut_beginning_data(data_nobaseline)
    Ncond, Npeople = X.shape
    N,D = X[0,0].shape  # Size of a chain
    for i in range(Npeople):
        X[0,i] = X[0,i]  # - np.mean(X[0,i], axis = 0).reshape(1,D)
#######################################################################################
######## Create the Distribution object ##############################################
#######################################################################################

#### 1st Gaussian Distribution
Gaussian_d = Cdist.CDistribution(name = "Gaussian");
Gaussian_d.set_distribution("Gaussian")
Gaussian_d.parameters["mu_variance"]  = 1e-100
Gaussian_d.parameters["Sigma_min_init"] = 1e-13
Gaussian_d.parameters["Sigma_max_init"] = 1e-14
Gaussian_d.parameters["Sigma_min_singularity"] = 0.1
Gaussian_d.parameters["Sigma_min_pdf"] = 0.1
Gaussian_d.parameters["Sigma"] = "full"

#### 1st Gaussian Distribution
Gaussian_d2 = Cdist.CDistribution(name = "Gaussian2");
Gaussian_d2.set_distribution("Gaussian")
Gaussian_d2.parameters["mu_variance"]  = 1e-100
Gaussian_d2.parameters["Sigma_min_init"] = 1e-13
Gaussian_d2.parameters["Sigma_max_init"] = 1e-14
Gaussian_d2.parameters["Sigma_min_singularity"] = 0.1
Gaussian_d2.parameters["Sigma_min_pdf"] = 0.1
Gaussian_d2.parameters["Sigma"] = "diagonal"

#### 2nd: Watson distribution
Watson_d = Cdist.CDistribution(name = "Watson");
Watson_d.set_distribution("Watson")
Watson_d.parameters["Num_Newton_iterations"] = 5
Watson_d.parameters["Allow_negative_kappa"] = "no"
Watson_d.parameters["Kappa_max_init"] = 100
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
### Add them together in the Manager
myDManager = Cdist.CDistributionManager()
K_G = 0   # Number of clusters for the Gaussian Distribution
K_G2 = 4  # Number of clusters for the Gaussian Distribution
K_W = 0
K_vMF = 0
if (K_G > 0):
    myDManager.add_distribution(Gaussian_d, Kd_list = range(0,K_G))
if(K_W > 0):
    myDManager.add_distribution(Watson_d, Kd_list = range(K_G,K_W+ K_G))
if(K_vMF > 0):
    myDManager.add_distribution(vonMisesFisher_d, Kd_list = range(K_W+ K_G,K_W+ K_G+ K_vMF))
if (K_G2 > 0):
    myDManager.add_distribution(Gaussian_d2, Kd_list = range(K_W+ K_G+ K_vMF,K_W+ K_G+ K_vMF+K_G2))

########################################################################################
#####################   Crossvalidate Using EM   ####################################
#########################################################################################
if (EM_evolution):
    ## Transform HMM lists in to a np.matrix (Nsam, Ndim) for the EM
    ### Number of clusters to crossvalidate
    Nfolds =  Npeople  # Number of folds of each crossvalidation
    
    ############# SET TRAINNG PARAMETERS ##################################
    Ninit = 3
    delta_ll = 0.02
    T = 30
    verbose = 2;
    clusters_relation = "independent"   # MarkovChain1  independent
    time_profiling = None
    
    K = len(myDManager.clusterk_to_Dname.keys())
    ### Data structures to save the results
    logl_tr_CVs = []  # List of the training LL of all CVs   [nCVs]
    logl_val_CVs = [] # List of the validation LL of all CVs   [nCVs]
    

    """ Now we perform the CV of the number of clusters just for one condition """
    
    cond_i = 0
    Xdata = list(X[cond_i,:])
    ############################# Begin CV  ####################################
    fake_labels = np.ones(len(Xdata)).flatten() # Emulation of all samples have the same class

    stkfold = cross_validation.StratifiedKFold(fake_labels, n_folds = Nfolds)
    for train_index, val_index in stkfold:  # For each partition, using the set of parameters to CV
        # Create the EM object with the hyperparameters
        
        Xdata_train = np.concatenate( [Xdata[indx] for indx in train_index], axis = 0)
        Xdata_val = np.concatenate( [Xdata[indx] for indx in val_index], axis = 0)
        
        myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
        T = T, Ninit = Ninit,  delta_ll = delta_ll, 
        verbose = verbose, time_profiling = time_profiling)
        ## Perform EM with the hyperparameters already set, we just fit the data and position init
        theta_init = None; model_theta_init = None
        ############# PERFORM THE EM #############
        logl,theta_list,mode_theta_list = myEM.fit(Xdata_train, model_theta_init = model_theta_init, theta_init = theta_init) 
        
        ### If we want to improve with the EM !
        
#        logl,theta_list,mode_theta_list = myEM.fit(Xdata_train, model_theta_init = model_theta_init, theta_init = theta_init) 
        break


    ##################  PLOT THE LIKELIHOOD EVOLUTION ####################
    gl.plot(range(1,np.array(logl).flatten()[1:].size +1),np.array(logl).flatten()[1:], 
            legend = ["EM LogLikelihood"], 
    labels = ["Convergence of LL with generated data","Iterations","LL"], lw = 2)


    perform_HMM_after_EM = 1
    if(perform_HMM_after_EM):
        Ninit = 1
        ############# Create the EM object and fit the data to it. #############
        clusters_relation = "MarkovChain1"   # MarkovChain1  independent
        myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
                       T = T, Ninit = Ninit,  delta_ll = delta_ll, 
                       verbose = verbose, time_profiling = time_profiling)
    
        if (0):
            theta_init = theta_list[-1]
            A_init = np.concatenate([mode_theta_list[-1][0] for k in range(K)], axis = 0)
            model_theta_init = [mode_theta_list[-1], A_init]
        else:
            theta_init = None
            model_theta_init = None
        ############# PERFORM THE EM #############
        logl,theta_list,mode_theta_list = myEM.fit(Xdata_train, model_theta_init = model_theta_init, theta_init = theta_init) 
        
        #######################################################################################################################
        #### Plot the evolution of the centroids likelihood ! #####################################################
        #######################################################################################################################
        
        gl.plot(range(1,np.array(logl).flatten()[1:].size +1),np.array(logl).flatten()[1:], 
                legend = ["HMM LogLikelihood"], 
        labels = ["Convergence of LL with generated data","Iterations","LL"], 
        lw = 2)
        gl.savefig(folder_images +'Likelihood_Evolution_EM_HMM. K_G:'+str(K_G)+ ', K_W:' + str(K_W) + ', K_vMF:' + str(K_vMF) + '.png', 
               dpi = 100, sizeInches = [12, 6])
        
    for k_c in myDManager.clusterk_to_Dname.keys():
        k = myDManager.clusterk_to_thetak[k_c]
        distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
        if (distribution_name == "Gaussian"):
            print ("------------ Gaussian Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[-1][k][0])
            print ("Sigma")
            print (theta_list[-1][k][1])
        elif(distribution_name == "Watson"):
            print ("------------ Watson Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[-1][k][0])
            print ("Kappa")
            print (theta_list[-1][k][1])
        elif(distribution_name == "vonMisesFisher"):
            print ("------------ vonMisesFisher Cluster. K = %i--------------------"%k)
            print ("mu")
            print (theta_list[-1][k][0])
            print ("Kappa")
            print (theta_list[-1][k][1])
    print ("model_theta")
    print (mode_theta_list[-1])
    
    
################################################################################################################
################################### PLOTTING THE RESULTS #####################################################
################################################################################################################
### Reschedule the data. We are going to for a single CV:
# Compute the mean LL of the CVs for train and validation


if (plot_time_stuff):
    
    new_ll = myEM.get_loglikelihood(Xdata,myDManager, theta_list[-1],mode_theta_list[-1])
    r = myEM.get_responsibilities(Xdata,myDManager, theta_list[-1],mode_theta_list[-1])
    
    if (myEM.clusters_relation == "independent"):
        Nsam, K = r.shape
    elif(myEM.clusters_relation == "MarkovChain1"):
        Nsam, K = r[0].shape
    Npeople_plot = Npeople
    legend = [" K = %i"%i for i in range(K)]
    
    gl.set_subplots(Npeople,1);
    ax1 = None
    legend = []
    
    labels_title = "Cluster responsibility for the EM"
    Ndiv = 4
    
    for i in range(Npeople_plot):
        if (myEM.clusters_relation == "independent"):
           resp = r[N*i:N*(i+1),:]
        elif(myEM.clusters_relation == "MarkovChain1"):
            resp = r[i]
            
        Nclusters = resp.shape[1]
        ax_ii = gl.subplot2grid((Npeople_plot,Ndiv), (i,0), rowspan=1, colspan=Ndiv-1) 
        ax1 = gl.plot_filled(time,resp , nf = 0, fill_mode = "stacked", legend = legend, 
                             sharex = ax1, sharey = ax1, AxesStyle = "Normal - No yaxis", labels = [labels_title,"","%i"%i])
        gl.colorIndex = 0
        labels_title = ""
        
    ax_i = gl.subplot2grid((Npeople_plot,Ndiv), (0,Ndiv-1), rowspan=Npeople_plot/2, colspan=1) 
    for i in range(1,K+1):
        gl.scatter(0, i, legend = [" K = %i"%(i)], lw = 28, AxesStyle = "Normal - No xaxis - No yaxis" , loc = "center left")
        
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
    image_name = "EM_all_subjects_trained_togehter"
#    
    gl.set_fontSizes(ax = ax_i, title = 20, xlabel = 20, ylabel = 20, 
                      legend = 35, xticks = 25, yticks = 10)
    gl.set_fontSizes(ax = ax_ii, title = 20, xlabel = 20, ylabel = 20, 
                      legend = 30, xticks = 20, yticks = 10)
    gl.set_zoom(xlim = [10,10.50])
    
    gl.savefig(folder_images + image_name, 
               dpi = 100, sizeInches = [30, 12])

## Save to disk the clusters

    mus_kk = []
    for i in range(K):
        mus_kk.append(theta_list[-1][i][0])
    
    mus_kk = np.concatenate(mus_kk,axis = 1)
    
    
#    df = pd.DataFrame(mus_kk)
#    df.to_csv(folder_images + "file_path.csv")

    np.savetxt(folder_images + "clusters.csv", mus_kk, delimiter=",")
#plotting_func = 1
#if (plotting_func):
#    
#    gl.init_figure()
#    
#    Npeople_plot = 5
#    
#    legend = [" K = %i"%i for i in range(K)]
#    
#    ax1 = None
#    legend = []
#    
#    labels_title = "Value of the 70 channels for a subject"
#    for i in range(Npeople_plot):
#        ax1 = gl.plot(time, Xdata[0][:,:], nf = 0, legend = legend, 
#                             sharex = ax1, sharey = ax1, AxesStyle = "Normal2", labels = [labels_title,"t","Sensor value"])
#        gl.colorIndex = 0
#        labels_title = ""
#        
#    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
#    image_name = "70_channels"
#    
#    gl.savefig(folder_images + image_name, 
#               dpi = 100, sizeInches = [30, 8])
#    
    
    
    
    
    
