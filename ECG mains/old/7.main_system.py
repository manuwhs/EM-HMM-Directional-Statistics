
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

import data_preprocessing as dp

plt.close("all")
from sklearn.model_selection import train_test_split

################################################################
######## Load the dataset ! ##############################
###############################################################

X_All_labels, label_classes = dp.load_one_person_dataset(dataset_folder = "./dataset/", 
                                                      filename = 'face_scrambling_spm8proc_sub07.mat')

X_All_labels = [X_All_labels[0], X_All_labels[2]]
label_classes = [label_classes[0], label_classes[2]]

Nclasses = len(X_All_labels)
X_All_labels = dp.remove_timePoints_average(X_All_labels)

# X_All_labels[Nclass][Ntrials per class][Trial (Ntime x Ndim)]

################################################################
######## Loading in lists and preprocessing! ##################
###############################################################
max_trials = 100  # Number of trials per class
Ntrials, NtimeSamples, Ndim = X_All_labels[0].shape
possible_channels = range(Ndim)
Nchannels_chosen = 5  # We select this number of channels

channel_sel = np.random.permutation(possible_channels)[0:Nchannels_chosen].flatten().tolist()      # Randomize the array of index
#channel_sel = [20, 21, 22]

X_data_trials, X_data_labels = dp.preprocess_data_set (
                                X_All_labels, label_classes, 
                                max_trials = max_trials, channel_sel= channel_sel)

X_data_ave = dp.get_timeSeries_average_by_label(X_All_labels, channel_sel = channel_sel)
for i in range (len(X_data_ave)):  # Put the data in the sphere.
    X_data_ave[i] = gf.normalize_module(X_data_ave[i])


################# Separate in train and validation ############                    
X_train, X_test, y_train, y_test = train_test_split(X_data_trials, X_data_labels, test_size=0.50, random_state = 0, stratify = X_data_labels)

####################################################### 
######################### EM ########################### 
####################################################### 

EM_flag = 1

if (EM_flag):
    Ninit = 10
    K  =  6
    verbose = 0
    T  = 50
    
    Ks_params = []
    for i in range(Nclasses):
        
        X_train_class_i = [X_data_ave[i]]
        logl,theta_list,pimix_list = EMl.run_several_EM(X_train_class_i, K = K, delta = 0.1, T = T,
                                    Ninit = Ninit, verbose = verbose)
        Ks_params.append([pimix_list[-1],theta_list[-1]])
        
#        Ks_params[class][0]
    Likelihoods = dp.get_likelihoods_EM(X_train, Ks_params)
    accu_train = gf.accuracy(y_train, np.argmax(Likelihoods, axis = 1))
    #    print [y_train, np.argmax(Likelihoods, axis = 1)]
    print "Train Accuracy %f" %(accu_train)
    
    Likelihoods = dp.get_likelihoods_EM(X_test, Ks_params)
    accu_test = gf.accuracy(y_train, np.argmax(Likelihoods, axis = 1))
    
    #    print [y_test, np.argmax(Likelihoods, axis = 1)]
    print "Test Accuracy %f" %(accu_test)

    mus_0 = Ks_params[0][1][0].T
    mus_1 = Ks_params[1][1][0].T
    
    max_correlation_0 = np.max(np.abs(mus_0.dot(mus_1.T)), axis = 0)
    max_correlation_1 = np.max(np.abs(mus_0.dot(mus_1.T)), axis = 1)
    
    print max_correlation_0
    print max_correlation_1