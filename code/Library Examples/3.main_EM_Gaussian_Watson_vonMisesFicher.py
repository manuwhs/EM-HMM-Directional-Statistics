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

import Gaussian_distribution as Gad
import Gaussian_estimators as Gae
import Watson_distribution as Wad
import general_func as gf
import basicMathlib as bMA
import CEM as CEM 
import CDistribution as Cdist
import utilities_lib as ul

import specific_plotting_func as spf

plt.close("all")

folder = "./data/test_data/"
folder_HMM = "./data/HMM_data/"

folder_images = "../pics/Trapying/EM_HMM/"
ul.create_folder_if_needed(folder_images)

generate_Gaussian_data = 1
plot_original_data= 1

perform_EM = 1
plot_evolution = 1
plot_evolution_2 = 0
plot_evolution_video = 0
plot_evolution_ll_video = 0
################################################################
######## Load and combine 3 sets ##############################
###############################################################

if (generate_Gaussian_data):
    X1,X2,X3,Xdata, mu1,mu2,mu3, cov1,cov2, cov3 = spf.generate_gaussian_data(folder_images, plot_original_data, 
                                                                              N1 = 2000, N2 = 2000, N3 = 2000 )

    ## Now we define the parameters of the HMM
    
    if (1):
        pi = np.array([0.2,0.3,0.5])
        A = np.array([[0.4, 0.1, 0.5],
                      [0.4, 0.5, 0.1],
                      [0.7, 0.1, 0.2]])
        
        Nchains = 10  # Number of chains
        Nsamples = 50 + np.random.rand(Nchains) * 10  # Number of samples per chain
        Nsamples = Nsamples.astype(int)  # Number of samples per chain
        
        # For training
        data_index = gf.draw_HMM_indexes(pi, A, Nchains, Nsamples)
        data_chains = gf.draw_HMM_samples(data_index, [X1.T,X2.T,X3.T])
        Xdata = data_chains
    else:
        data_chains = [np.concatenate((X1,X2), axis = 1).T]
        Xdata = data_chains
#######################################################################################################################
######## Perform the EM !! ##############################################################################
#################################################################################################################

if (perform_EM):
#    N,D = Xdata.shape
    ######## Create the Distribution object ##############################################
    
    #### 1st Gaussian Distribution
    ## Create the distribution object and set it a Gassian.
    Gaussian_d = Cdist.CDistribution(name = "Gaussian");
    Gaussian_d.set_distribution("Gaussian")
    
    Gaussian_d.parameters["mu_variance"] = 1
    Gaussian_d.parameters["Sigma_min_init"] = 1
    Gaussian_d.parameters["Sigma_max_init"] = 15
    Gaussian_d.parameters["Sigma"] = "diagonal" #  "diagonal"    "full" 
    Gaussian_d.parameters["Sigma_min_singularity"] = 0.1
    Gaussian_d.parameters["Sigma_min_pdf"] = 0.1

    
    #### 2nd: Watson distribution
    Watson_d = Cdist.CDistribution(name = "Watson");
    Watson_d.set_distribution("Watson")
    
    Watson_d.use_changeOfClusters = None
    Watson_d.parameters["Num_Newton_iterations"] = 5
    Watson_d.parameters["Allow_negative_kappa"] = "no"
    Watson_d.parameters["Kappa_max_init"] = 100
    Watson_d.parameters["Kappa_max_singularity"] =1000
    Watson_d.parameters["Kappa_max_pdf"] = 1000
    
    #### 3rd von MisesFisher distribution ######
    vonMisesFisher_d = Cdist.CDistribution(name = "vonMisesFisher");
    vonMisesFisher_d.set_distribution("vonMisesFisher")
    
    vonMisesFisher_d.parameters["Num_Newton_iterations"] = 2
    vonMisesFisher_d.parameters["Kappa_max_init"] = 20
    vonMisesFisher_d.parameters["Kappa_max_singularity"] =1000
    vonMisesFisher_d.parameters["Kappa_max_pdf"] = 1000
    ### Add them together in the Manager
    myDManager = Cdist.CDistributionManager()
    K_G = 1      # Number of clusters for the Gaussian Distribution
    K_W =  2
    K_vMF = 0
    if (K_G > 0):
        myDManager.add_distribution(Gaussian_d, Kd_list = range(0,K_G))
    if(K_W > 0):
        myDManager.add_distribution(Watson_d, Kd_list = range(K_G,K_W+ K_G))
    if(K_vMF > 0):
        myDManager.add_distribution(vonMisesFisher_d, Kd_list = range(K_W+ K_G,K_W+ K_G+ K_vMF))
        
#    Gaussian_d.set_parameters(parameters)
    ############# SET TRAINNG PARAMETERS ##################################
    K = len(myDManager.clusterk_to_Dname.keys())
    Ninit = 10
    delta_ll = 0.02
    T = 20
    verbose = 1;
    clusters_relation = "independent"   # MarkovChain1  independent
    time_profiling = None
    
    
    ############# Create the EM object and fit the data to it. #############
    myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
                   T = T, Ninit = Ninit,  delta_ll = delta_ll, 
                   verbose = verbose, time_profiling = time_profiling)

    ### Set the initial parameters
    precompute_init = 0
    if (precompute_init):
        theta_init = Gad.init_params(K,D, theta_init = None, parameters = Gad.parameters)
        model_theta_init = EMl.init_model_params(K)
    else:
        theta_init = None
        model_theta_init = None
    ############# PERFORM THE EM #############
    
    logl,theta_list,mode_theta_list = myEM.fit(Xdata, model_theta_init = model_theta_init, theta_init = theta_init) 
    
    #######################################################################################################################
    #### Plot the evolution of the centroids likelihood ! #####################################################
    #######################################################################################################################
    
    gl.init_figure()
    gl.plot(range(1,np.array(logl).flatten()[1:].size +1),np.array(logl).flatten()[1:], 
            legend = ["EM LogLikelihood"], 
    labels = ["Convergence of LL with generated data","Iterations","LL"], 
    lw = 2)
    gl.savefig(folder_images +'Likelihood_Evolution. K_G:'+str(K_G)+ ', K_W:' + str(K_W) + ', K_vMF:' + str(K_vMF)+ '.png', 
           dpi = 100, sizeInches = [12, 6])

    perform_HMM_after_EM = 1
    if(perform_HMM_after_EM):
        Ninit = 1
        ############# Create the EM object and fit the data to it. #############
        clusters_relation = "MarkovChain1"   # MarkovChain1  independent
        myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
                       T = T, Ninit = Ninit,  delta_ll = delta_ll, 
                       verbose = verbose, time_profiling = time_profiling)
    
        theta_init = theta_list[-1]
        A_init = np.concatenate([mode_theta_list[-1][0] for k in range(K)], axis = 0)
        model_theta_init = [mode_theta_list[-1], A_init]
#        theta_init = None
#        model_theta_init = None
        ############# PERFORM THE EM #############
        Xdata = data_chains
        logl,theta_list,mode_theta_list = myEM.fit(Xdata, model_theta_init = model_theta_init, theta_init = theta_init) 
        
        #######################################################################################################################
        #### Plot the evolution of the centroids likelihood ! #####################################################
        #######################################################################################################################
        
        gl.plot(range(1,np.array(logl).flatten()[1:].size +1),np.array(logl).flatten()[1:], 
                legend = ["HMM LogLikelihood"], 
        labels = ["Convergence of LL with generated data","Iterations","LL"], 
        lw = 2)
        gl.savefig(folder_images +'Likelihood_Evolution. K_G:'+str(K_G)+ ', K_W:' + str(K_W) + ', K_vMF:' + str(K_vMF) + '.png', 
               dpi = 100, sizeInches = [12, 6])


#######################################################################################################################
#### Obtain the evolution of the centroids to plot them properly #####################################################
#######################################################################################################################

if(plot_evolution):
   spf.plot_final_distribution([X1,X2,X3],[mu1,mu2,mu3],[cov1,cov2,cov3], [K_G, K_W, K_vMF],myDManager, logl,theta_list,mode_theta_list,folder_images)

if (plot_evolution_2):
    spf.plot_multiple_iterations([X1,X2,X3],[mu1,mu2,mu3],[cov1,cov2,cov3], [K_G, K_W, K_vMF],myDManager, logl,theta_list,mode_theta_list,folder_images)

if (plot_evolution_video):
    folder_images_gif = "../pics/Trapying/EM_HMM/gif/"
    spf.generate_images_iterations([X1,X2,X3],[mu1,mu2,mu3],[cov1,cov2,cov3], [K_G, K_W, K_vMF],myDManager, logl,theta_list,mode_theta_list,folder_images_gif)
    
    #### Load the images 
    images_path = ul.get_allPaths(folder_images_gif, fullpath = "no")
    images_path.sort(cmp = ul.comparador_images_names)
    ### Create Gif ###
    output_file_gif = 'Evolution_gif. K_G:'+str(K_G)+ ', K_W:' + str(K_W) + ', K_vMF:' + str(K_vMF)+".gif"
    ul.create_gif(images_path,folder_images + output_file_gif, duration = 0.1)
    ## Create video ##
    output_file = folder_images + 'Evolution_video. K_G:'+str(K_G)+ ', K_W:' + str(K_W) + ', K_vMF:' + str(K_vMF) +'.avi'
    ul.create_video(images_path, output_file = output_file, fps = 5)
    
if (plot_evolution_ll_video):
    folder_images_gif = "../pics/Trapying/EM_HMM/gif/"
    spf.generate_images_iterations_ll([X1,X2,X3],[mu1,mu2,mu3],[cov1,cov2,cov3], [K_G, K_W, K_vMF],myDManager, logl,theta_list,mode_theta_list,folder_images_gif)
    #### Load the images 
    images_path = ul.get_allPaths(folder_images_gif, fullpath = "no")
    images_path.sort(cmp = ul.comparador_images_names)
    ## Create video ##
    output_file = folder_images + 'Evolution_video_ll. K_G:'+str(K_G)+ ', K_W:' + str(K_W) + ', K_vMF:' + str(K_vMF)+'.avi'
    ul.create_video(images_path, output_file = output_file, fps = 5)


#r = myEM.get_responsibilities(Xdata,myDManager, theta_list[-1],mode_theta_list[-1])
#rmax = np.argmax(r, axis = 1)
#N,K = r.shape
#hard_r = np.zeros(r.shape)
#for i in range(N):
#    hard_r[i,rmax[i]] = 1