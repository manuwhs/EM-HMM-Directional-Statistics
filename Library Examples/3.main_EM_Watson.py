"""
This code will:
        - Load previously generated data in 3D for either EM and HMM.
        - Perform the EM
        - Plot the data and the evolution of the clusters.
        - Plot the evolution of LL with the iterations of EM
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

load_EM_data = 0
load_HMM_data = 1
perform_EM = 1
################################################################
######## Load and combine 3 sets ##############################
###############################################################

if (load_EM_data):
    K = 3
    #gl.scatter_3D([0,1,1,1,1,-1,-1,-1,-1], [0,1,1,-1,-1,1,1,-1,-1],[0,1,-1,1,-1,1,-1,1,-1], nf = 1, na = 0)
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    kflag = 0
    for k in range(1,K+1):

        filedir = folder + "Wdata_"+ str(k)+".csv"
        Xdata_k = np.array(pd.read_csv(filedir, sep = ",", header = None))
        Xdata_k = Xdata_k[:1000,:]
    #    Xdata_param = pkl.load_pickle( folder + "Wdata_"+ str(k)+".pkl",1)
    #    mu = Xdata_param[0]
    #    kappa = Xdata_param[1]
        
    #    print "Real: ", mu,kappa
        # Generate and plot the data
        
        gl.scatter_3D(Xdata_k[:,0], Xdata_k[:,1],Xdata_k[:,2], nf = 0, na = 0)
        
        mu_est2, kappa_est2 = Wae.get_Watson_muKappa_ML(Xdata_k)
        print (["ReEstimated: ", mu_est2.T,kappa_est2])
        
        if (kflag == 0):
            Xdata = copy.deepcopy(Xdata_k)
            kflag = 1
        else:
            Xdata = np.concatenate((Xdata, copy.deepcopy(Xdata_k)), axis = 0)

################################################################
######## Or Load the same data as for HMM 3 sets ###############
###############################################################

if (load_HMM_data):
    HMM_list = pkl.load_pickle(folder_HMM +"HMM_datapoints.pkl",1)
    
#    gl.scatter_3D([0,1,1,1,1,-1,-1,-1,-1], [0,1,1,-1,-1,1,1,-1,-1],[0,1,-1,1,-1,1,-1,1,-1], nf = 1, na = 0)
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    k = 0 # For the initial
    for Xdata_chain in HMM_list:
        Xdata_chain = np.array(Xdata_chain)
        gl.scatter_3D(Xdata_chain[:,0], Xdata_chain[:,1],Xdata_chain[:,2], nf = 0, na = 0, color = "k", alpha = 0.4)
    Xdata = HMM_list

#######################################################################################################################
######## Perform the EM !! ##############################################################################
#################################################################################################################

if (perform_EM):
    #######################################################################################
    ######## Create the Distribution object ##############################################
    #######################################################################################
    
    #### 1st Gaussian Distribution
    Gaussian_d = Cdist.CDistribution(name = "Gaussian");
    Gaussian_d.set_distribution("Gaussian")
    #### 2nd: Watson distribution
    Watson_d = Cdist.CDistribution(name = "Watson");
    Watson_d.set_distribution("Watson")
    Watson_d.parameters["Allow_negative_kappa"] = "yes"
    
    ### Add them together in the Manager
    myDManager = Cdist.CDistributionManager()
    K_G = 0       # Number of clusters for the Gaussian Distribution
    K_W = 3
    if (K_G > 0):
        myDManager.add_distribution(Gaussian_d, Kd_list = range(0,K_G))
    if(K_W > 0):
        myDManager.add_distribution(Watson_d, Kd_list = range(K_G,K_W+ K_G))

#    Gaussian_d.set_parameters(parameters)
    ############# SET TRAINNG PARAMETERS ##################################
    K = len(myDManager.clusterk_to_Dname.keys())
    Ninit = 5
    delta_ll = 0.02
    T = 60
    verbose = 1;

    clusters_relation = "MarkovChain1"   # MarkovChain1  independent
    time_profiling = None
    
    
    ############# Create the EM object and fit the data to it. #############
    myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
                   T = T, Ninit = Ninit,  delta_ll = delta_ll, 
                   verbose = verbose, time_profiling = time_profiling)

    ### Set the initial parameters

    theta_init = None
    model_theta_init = None
    ############# PERFORM THE EM #############
    logl,theta_list,mode_theta_list = myEM.fit(Xdata, model_theta_init = model_theta_init, theta_init = theta_init) 
    
    #######################################################################################################################
    ## Process the results !!!
    
    mus_Watson_Gaussian = []
    # k_c is the number of the cluster inside the Manager. k is the index in theta
    for k_c in myDManager.clusterk_to_Dname.keys():
        k = myDManager.clusterk_to_thetak[k_c]
        distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
        mus_k = []
        for iter_i in range(len(theta_list)): # For each iteration of the algorihtm
            if (distribution_name == "Gaussian"):
                theta_i = theta_list[iter_i][k]
                mus_k.append(theta_i[0])
            elif(distribution_name == "Watson"):
                theta_i = theta_list[iter_i][k]
                mus_k.append(theta_i[0])
        
        mus_k = np.concatenate(mus_k, axis = 1).T
        mus_Watson_Gaussian.append(mus_k)
    
    
    plot_evolution = 1
    if (plot_evolution):
        # Only doable if the clusters dont die
        Nit,Ndim = mus_Watson_Gaussian[0].shape
        for k_c in myDManager.clusterk_to_Dname.keys():
            k = myDManager.clusterk_to_thetak[k_c]
            distribution_name = myDManager.clusterk_to_Dname[k_c] # G W
            
            gl.scatter_3D(mus_Watson_Gaussian[k][:,0],mus_Watson_Gaussian[k][:,1],mus_Watson_Gaussian[k][:,2], nf = 0, na = 0, join_points = "yes", alpha = 0.1,
                          color = "red")
        
    gl.init_figure()
    gl.plot(range(1,np.array(logl).flatten()[1:].size +1),np.array(logl).flatten()[1:], 
            legend = ["EM LogLikelihood"], 
    labels = ["Convergence of LL with generated data","Iterations","LL"], 
    lw = 2)


#gl.plot(range(1,np.array(caca).flatten()[1:].size +1),np.array(caca).flatten()[1:], nf = 0,
#        legend = ["HMM LogLikelihood"], 
#labels = ["Convergence of LL with generated data","Iterations","LL"], 
#lw = 4,
#        fontsize = 25,   # The font for the labels in the title
#        fontsizeL = 30,  # The font for the labels in the legeng
#        fontsizeA = 20)
#        
#Sel = EMlf.EMdecode(Xdata,[mus_list[-1], kappas_list[-1]], pimix_list[-1])

