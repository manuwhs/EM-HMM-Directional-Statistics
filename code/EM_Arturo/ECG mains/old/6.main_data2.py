
# Official libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
# Own libraries
import import_folders
from graph_lib import gl


import sampler_lib as sl
import EM_lib as EMl
import EM_libfunc as EMlf
import HMM_lib as HMMl
import HMM_libfunc2 as HMMlf
import decoder_lib as decl
import pickle_lib as pkl
import scipy.io
from sklearn import preprocessing

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf

plt.close("all")

## TODO: Idea, just pick 3 at random several times :) And get the likelihoods.

################################################################
######## Load the dataset ! ##############################
###############################################################
dataset_folder = "./dataset/"
mat = scipy.io.loadmat(dataset_folder +'face_scrambling_spm8proc_sub07.mat')
keys = mat.keys()
print keys
X = mat["X"]   # Nchannels x Time x Ntrials
trial_indices = mat["trial_indices"][0][0]  # Labels of the trials
label_classes = ["Famous", "Unfamiliar", "Scrambled"]

X_All_labels = []  # Contains the trials for every label
for label in label_classes:
    label_trials = trial_indices[label].flatten()
    X_label_trials = X[:,:,np.where(label_trials == 1)[0]].T
    X_All_labels.append(X_label_trials)

X_All_labels = [X_All_labels[0], X_All_labels[2]]
label_classes = [label_classes[0], label_classes[2]]

# Now X_All_labels has in every postion the trials for each class in the form
# of a matrix Ntrials x Ntimes x Ndim


################################################################
######## Preprocessing ! ##############################
###############################################################
# For the first label

plotting_all_trials_one_instant = 0

if (plotting_all_trials_one_instant):
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    Ntrial, Nsam, Ndim = X_All_labels[0].shape
    i0 = 50  # Good one !!
    
    t_show = 30
    for i in range (Ntrial):
        X_trial = X_All_labels[0][i,:,i0:i0+3] # Nsam x Ndim
        
    #    X_trial = X_trial - np.sum(X_trial, axis = 1).reshape(X_trial.shape[0],1)
    #    scaler = preprocessing.StandardScaler().fit(X_trial)
    #    X_trial = scaler.transform(X_trial)            
        
        X_trial = gf.normalize_data(X_trial)
        gl.scatter_3D(X_trial[t_show,0], X_trial[t_show,1],X_trial[t_show,2], nf = 0, na = 0)
     

#K  =  2
#logl,theta_list,pimix_list = EMl.run_several_EM(X_trial_ave, K = K, delta = 0.1, T = 100,
#                            Ninit = 1)
#mus_list = []
#kappas_list =[]
#for theta in theta_list:
#    mus_list.append(theta[0])
#    kappas_list.append(theta[1]) 
#print "kappas"
#print kappas_list[-1]
#print "mus"
#print mus_list[-1]
#print "pimix"
#print pimix_list[-1]

#K  =  2
#logl,theta_list,pimix_list = EMl.run_several_EM(X_trial, K = K, delta = 0.1, T = 100,
#                            Ninit = 1)
#mus_list = []
#kappas_list =[]
#for theta in theta_list:
#    mus_list.append(theta[0])
#    kappas_list.append(theta[1]) 
#print "kappas"
#print kappas_list[-1]
#print "mus"
#print mus_list[-1]
#print "pimix"
#print pimix_list[-1]

plotting_ave_trial_time = 0

if (plotting_ave_trial_time):
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    
    X_trial_ave = np.sum(X_All_labels[0][:,:,i0:i0+3] , axis = 0)
    X_trial_ave = gf.normalize_data(X_trial_ave)
    gl.scatter_3D(X_trial_ave[:,0], X_trial_ave[:,1],X_trial_ave[:,2], nf = 0, na = 0)
    
    X_trial_ave = np.sum(X_All_labels[1][:,:,i0:i0+3] , axis = 0)
    X_trial_ave = gf.normalize_data(X_trial_ave)
    gl.scatter_3D(X_trial_ave[:,0], X_trial_ave[:,1],X_trial_ave[:,2], nf = 0, na = 0)
    
    X_trial_ave = np.sum(X_All_labels[0][:,:,i0:i0+3] , axis = 0)
    X_trial_ave = gf.normalize_data(X_trial_ave)
    gl.plot([],X_trial_ave, color = "k")
    
    X_trial_ave = np.sum(X_All_labels[1][:,:,i0:i0+3] , axis = 0)
    X_trial_ave = gf.normalize_data(X_trial_ave)
    gl.plot([],X_trial_ave, nf = 0,color = "b")

plotting_trial_time = 1
if (plotting_trial_time):
    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    colors = ["k","r"]
    
    for label_i in range(2):
        X_ave = np.mean(X_All_labels[label_i][:,:,i0:i0+3],axis = 0)
        mean_shit = np.mean(X_ave, axis = 0).reshape(1,3)
        for tr_i in range (20):
            
            X_trial_ave = np.mean(X_All_labels[label_i][[tr_i],:,i0:i0+3] , axis = 0)
            X_trial_ave = X_trial_ave - mean_shit
            X_trial_ave = gf.normalize_data(X_trial_ave)
            
            gl.scatter_3D(X_trial_ave[:,0], X_trial_ave[:,1],X_trial_ave[:,2], nf = 0, na = 0,color = colors[label_i])

    gl.plot([0,0],[0,0], color = "k", nf = 1)
    

    for label_i in range(2):
        X_ave = np.mean(X_All_labels[label_i][:,:,i0:i0+3],axis = 0)
        mean_shit = np.mean(X_ave, axis = 0).reshape(1,3)
        for tr_i in range (20):
            X_trial_ave = np.mean(X_All_labels[label_i][[tr_i],:,i0:i0+3] , axis = 0)
            X_trial_ave = X_trial_ave - mean_shit
            X_trial_ave = gf.normalize_data(X_trial_ave)
            
            gl.plot([],X_trial_ave, color = colors[label_i], nf = 0)

    gl.plot([0,0],[0,0], color = "k", nf = 1)
#    gl.scatter_3D(0, 0,0, nf = 1, na = 0)
    
    for label_i in range(2):
        X_ave = np.mean(X_All_labels[label_i][:,:,i0:i0+3],axis = 0)
        mean_shit = np.mean(X_ave, axis = 0).reshape(1,3)
        std_shit = np.std(X_ave, axis = 0).reshape(1,3)

        print mean_shit
        print std_shit
        for tr_i in range (100):
            
            X_trial = np.mean(X_All_labels[label_i][[tr_i],:,i0:i0+3] , axis = 0)
            X_trial = X_trial  # - X_ave
#            X_ave = gf.normalize_data(X_ave)
            
#            gl.scatter_3D(np.max(X_trial[:,0]), np.max(X_trial[:,1]),np.max(X_trial[:,2]), nf = 0, na = 0,color = colors[label_i])
#            gl.scatter_3D(X_trial[:,0], X_trial[:,1],X_trial[:,2], nf = 0, na = 0,color = colors[label_i])
#            gl.scatter_3D(X_ave[:,0], X_ave[:,1],X_ave[:,2], nf = 0, na = 0,color = colors[label_i])
#            X_trial_ave = gf.normalize_data(X_trial_ave)
            
            gl.plot([],X_ave[:,0], color = colors[label_i], nf = 0)
            gl.plot([],X_ave[:,0] - std_shit[0][0], color = colors[label_i], nf = 0)
            gl.plot([],X_ave[:,0] + std_shit[0][0], color = colors[label_i], nf = 0)
            
#gl.scatter_3D(X_trial[:,0], X_trial[:,1],X_trial[:,2], nf = 1, na = 0)
 
#mu_est2, kappa_est2 = Wae.get_Watson_muKappa_ML(X_trial)
#print "Estimates ", mu_est2, kappa_est2

####################################################### 
######################### EM ########################### 
####################################################### 
#K  =  2
#logl,theta_list,pimix_list = EMl.run_several_EM(X_trial, K = K, delta = 0.1, T = 100,
#                            Ninit = 1)
#mus_list = []
#kappas_list =[]
#for theta in theta_list:
#    mus_list.append(theta[0])
#    kappas_list.append(theta[1]) 
#    
#plot_evolution = 1
#if (plot_evolution):
#    # Only doable if the clusters dont die
#    mus_array = np.array(mus_list) # (Nit,Ndim,Nk)
#    Nit,Ndim,Nk = mus_array.shape
#    for k in range(Nk):
#        gl.scatter_3D(mus_array[:,0,k], mus_array[:,1,k],mus_array[:,2,k], nf = 0, na = 0, join_points = "yes", alpha = 0.1)
#    
#gl.plot([],np.array(logl[1:]).flatten())
#
#print "kappas"
#print kappas_list[-1]
#print "mus"
#print mus_list[-1]
#print "pimix"
#print pimix_list[-1]

