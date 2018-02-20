"""
This code will generate the code to draw samples from the Watson, generate EM and HMM data, check the estimated parameters and save 
the generated data to disk.
"""

# Change main directory to the main folder and import folders
import os
os.chdir("../")
import import_folders
# Classical Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Own libraries
from graph_lib import gl
import sampler_lib as sl
import EM_lib as EMl
import pickle_lib as pkl

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf

plt.close("all")

################################################################
######## Generate and save 1 Dataset ###########################
################################################################
folder_EM = "../data/EM_data/"
folder_HMM = "../data/HMM_data/"

generate_data = 1

if (generate_data):
    ### Generate the data, save it to file, load it and estimate parameters
    kappa = 20
    N = 1000
    mu = np.array([2,4,5])
    mu = mu / np.sqrt(np.sum(mu * mu))
    
    RandWatson = Was.randWatson(N, mu, kappa)
    gl.scatter_3D(RandWatson[:,0], RandWatson[:,1],RandWatson[:,2], nf = 1, na = 0)
    
    print "Generation Parameters" 
    print mu
    print kappa
    
    print "Estimation Parameters" 
    mu = Wae.get_MLmean(RandWatson)
    kappa = Wae.get_MLkappa(mu, RandWatson)
    print mu
    print kappa
    
    ## Save the file to disk !!
    filedir = folder_EM +"Wdata.csv"
    np.savetxt(filedir, RandWatson, delimiter = ",")

    ##########################################
    ## Read the dataset and estimate the parameters
    Xdata = np.array(pd.read_csv(filedir, sep = ",", header = None))
    gl.scatter_3D(Xdata[:,0], Xdata[:,1],Xdata[:,2], nf = 0, na = 0)
    
    mu = Wae.get_MLmean(Xdata)
    kappa = Wae.get_MLkappa(mu, Xdata)
    
    print "Estimation Parameters after reading" 
    print mu
    print kappa

################################################################
######## Generate and save N Datasets ###########################
################################################################

K = 3
gl.scatter_3D(0, 0,0, nf = 1, na = 0)
N = 10000

Xdata = []  # List will all the generated data

for k in range(K):
    mu = np.random.uniform(-1,1,(1,3)).flatten()
    mu = mu / np.sqrt(np.sum(mu * mu))
    kappa = 50 + np.random.rand(1) * 50
    kappa = - kappa
    print "Real: ", mu,kappa
    
    filedir = folder_EM + "Wdata_"+ str(k)+ ".csv"

    # Generate and plot the data
    Xdata_k = Was.randWatson(N, mu, kappa)
    gl.scatter_3D(Xdata_k[:,0], Xdata_k[:,1],Xdata_k[:,2], nf = 0, na = 0)
    
    # pickle the parameters 
    pkl.store_pickle(folder_EM +"Wdata_"+ str(k)+".pkl",[mu,kappa],1)

    # Print an estimation of the parameters

    mu = Wae.get_MLmean(Xdata_k)
    kappa = Wae.get_MLkappa(mu, Xdata_k)
    
    print "Estimated: ", mu,kappa
    np.savetxt(filedir, Xdata_k, delimiter = ",")

    Xdata.append(Xdata_k)
    
################################################################
######## Generate HMM data and store it ###########################
################################################################

## Now we define the parameters of the HMM
pi = np.array([0.2,0.3,0.5])
A = np.array([[0.4, 0.1, 0.5],
              [0.4, 0.5, 0.1],
              [0.7, 0.1, 0.2]])

## For every chain, we draw a sample according to pi and then
## We keep drawing samples according to A

Nchains = 20  # Number of chains
Nsamples = 100 + np.random.rand(Nchains) * 100  # Number of samples per chain
Nsamples = Nsamples.astype(int)  # Number of samples per chain

# For training
Chains_list = gf.draw_HMM_indexes(pi, A, Nchains, Nsamples)
HMM_list = gf.draw_HMM_samples(Chains_list, Xdata)

## For validation !!!
Chains_list2 = gf.draw_HMM_indexes(pi, A, Nchains, Nsamples)
HMM_list2 = gf.draw_HMM_samples(Chains_list2, Xdata)

gl.scatter_3D(0, 0, 0, nf = 1, na = 0)

for XdataChain in HMM_list:
    gl.scatter_3D(XdataChain[:,0], XdataChain[:,1],XdataChain[:,2], nf = 0, na = 0)


# We pickle the information
# This way we have the same samples for EM and HMM

pkl.store_pickle(folder_HMM +"HMM_labels.pkl",Chains_list,1)
pkl.store_pickle(folder_HMM +"HMM_datapoints.pkl",HMM_list,1)
pkl.store_pickle(folder_HMM +"HMM_param.pkl",[pi,A],1)

pkl.store_pickle(folder_HMM +"HMM2_datapoints.pkl",HMM_list2,1)
#Xdata = np.concatenate((Xdata1,Xdata2,Xdata3), axis = 0)

