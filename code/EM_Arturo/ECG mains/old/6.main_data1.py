
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
import system_modules as sm
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
plt.close("all")


######## Load the dataset ! ##############################
load_dataset = 1
if (load_dataset):
    print "Loading Data"
    filename =  'face_scrambling_spm8proc_sub07.mat'
    X_All_labels, label_classes = dp.load_one_person_dataset(dataset_folder = "./dataset/", 
                                                          filename = filename)
    # Erase the middle class
    X_All_labels = [X_All_labels[0], X_All_labels[2]]
    label_classes = [label_classes[0], label_classes[2]]
    Nclasses = len(X_All_labels)
# X_All_labels[Nclass][Ntrials per class][Trial (Ntime x Ndim)]

######## Loading in lists and preprocessing! ##################
creating_data = 1
if (creating_data):
    print "Preprocessing Data"
    channel_sel = range(70)
#    channel_sel = [20, 21, 22, 0, 1, 2, 31,32,34]
    max_trials = 300
    
    X_data_trials, X_data_labels = sm.create_data(X_All_labels, label_classes, 
                channel_sel = channel_sel, max_trials = max_trials,
                rem_timePoint_ave = True, rem_features_ave = False, 
                normalize = False )

    ##### Separate in train and validation  
    r_seed = np.abs(int(100 * np.random.randn()))   
    X_train_notNormalized, X_test_notNormalized, y_train, y_test = train_test_split(X_data_trials, X_data_labels, test_size=0.5, random_state = r_seed, stratify = X_data_labels)

    CV_flag = 0
    if (CV_flag):
        # Example on how to do crossvalidation
        stkfold = cross_validation.StratifiedKFold(X_data_labels, n_folds = 5)
        for train_index, val_index in stkfold:
            X_train_notNormalized = [X_data_trials[itr] for itr in train_index]
            y_train = [X_data_labels[itr] for itr in train_index]
            
            X_test_notNormalized = [X_data_trials[iv] for iv in val_index]
            y_test = [X_data_labels[iv] for iv in val_index]

            # XXXXXXXX  REST OF THE PROCESSING ##########

    # If we want some PCA :)
#    n_components = 10
#    X_train_notNormalized, X_test_notNormalized = sm.pca_decomp(X_train_notNormalized, X_test_notNormalized,
#                                                                n_components = n_components)
    X_train = dp.normalize_trialList(X_train_notNormalized)
    X_test = dp.normalize_trialList(X_test_notNormalized)
    
plots_flag = 0
if (plots_flag):
    # Just some functions that you can call to vizualize stuff bitch !
#    sm.plot_PCA_example(X_train, y_train)
#    sm.plot_PCA_example(X_train_notNormalized, y_train)
    sm.plot_means(Nclasses, X_train, y_train,  # X_train_notNormalized
               colors = ["r","k"], normalize = False) 
#               
#    sm.plot_trials_for_same_instance(X_data_trials, X_data_labels, X_train, y_train,
#                                  colors = ["r","k"], time_show = 102, normalize = True)
#    sm.plot_single_trials(Nclasses, X_train_notNormalized, y_train,
#                            n_trials_to_show = 20 , colors = ["r","k"])

#    sm.plot_data_trials(Nclasses, X_train, y_train,
#                        n_trials_to_show = 2 , colors = ["r","k"])

## TODO: Try-cath general del EM por si crashea.
## TODO: Impove time in estimation of Kappa. Precompute the Watson probablities
######################### EM ########################### 
EM_flag = 0
if (EM_flag):
    print "Performing EM"
    ## We feed it with the average of the X_train, before normalizing
    ## Create mean profile only with training samples !
    X_data_ave = dp.get_average_from_train(Nclasses, X_train_notNormalized, y_train, 
                                           normalize = True, partitions = 1)
    Ninit = 5; K  =  3; verbose = 0; T  = 200
    Ks_params = sm.get_clusters_labels_EM(Nclasses, X_train = X_data_ave, y_train = range(Nclasses), 
                             Ninit = Ninit, K  =  K, T  = T, 
                             verbose = verbose)
    print "Finished EM"

CV_EM = 0
if (CV_EM):
    # We compute the EM for both clusters for the classes.
    # We do it for different kluster sizes K. We train with the means of train and we validate with the means of test.
    Klusters = [1,2,3,4,5,6,7,9,10,12,14,18,20,22,24] # ,13,15,17,20,25
    
    # For time reasons, we just perform the EM CV a lot of times with only one init
    # and then we keep drawing the graph or something when finished.
    N_init_CV = 50
    ll_train_best, ll_test_best, All_Ks_params_best = [], [], []
    for i in range(N_init_CV):
        ll_train, ll_test, All_Ks_params = sm.perfrom_CV_EM_classes(Nclasses, X_data_trials, X_data_labels, 
                              Klusters = Klusters, Ninit = 10, T = 100, nfolds = 5)
        if (i == 0):
            ll_train_best, ll_test_best, All_Ks_params_best  = ll_train, ll_test, All_Ks_params
        else: 
            for K_i in range (len(Klusters)):
                for ic in range(Nclasses):
                    if (ll_train[ic,K_i] > ll_train_best[ic,K_i]):
                        ll_train_best[ic,K_i] = copy.deepcopy(ll_train[ic,K_i])
                        ll_test_best[ic,K_i] = copy.deepcopy(ll_test[ic,K_i])
                        All_Ks_params_best[K_i] = copy.deepcopy(All_Ks_params[K_i])
        
        for ic in range(Nclasses):
            gl.plot(Klusters, np.array([ll_train_best[ic], ll_test_best[ic]]).T, legend = ["tr", "Val"], labels = ["EM class = " + str(ic),"States","loglike"])
            gl.savefig(file_dir = "./OnePerson_5fold_cluster" + str(ic) + "/Iteration" +str(i)+  ".png", 
            bbox_inches = 'tight',
            sizeInches = [],  # The size in inches as a list
            close = True,   # If we close the figure once saved
            dpi = 100)      # Density of pixels !! Same image but more cuality ! Pixels

loading_precomputed_centroids = 1
if (loading_precomputed_centroids):
#    pkl.store_pickle("./OnePerson1FoldEM.pkl",[ll_train_best, ll_test_best, All_Ks_params_best])
    cosas = pkl.load_pickle("./OnePerson1FoldEM.pkl")
    class_i = 1;
    n_cluster_opt = 5;
    good_clusters_EM = cosas[2][n_cluster_opt][class_i]
    Ks_params = good_clusters_EM
    pi_opt = good_clusters_EM[0]
    mu_opt = good_clusters_EM[1][0]
    
    Ks_params = cosas[2][n_cluster_opt]
#    cosas = pkl.load_pickle("./OnePerson1FoldHMM.pkl")
#    class_i = 1;
#    n_cluster_opt = 3;
#    good_clusters_EM = cosas[2][n_cluster_opt][class_i]
#    Ks_params_HMM = good_clusters_EM
#    pix_opt = good_clusters_EM[0]
#    A_opt = good_clusters_EM[1]
#    B_opt = good_clusters_EM[2]
#    mu_opt = B_opt[0]

#    np.savetxt('clusters_HMM_class_scrambled.csv', mu_opt.T, delimiter=',')   # X is an array

EM_evolution = 0
if (EM_evolution):
    print "Performing 1 EM to plot the time evolution of likelihood"
    X_data_ave = dp.get_average_from_train(Nclasses, X_train_notNormalized, y_train, 
                                           normalize = True, partitions = 1)
    ## RUN only one !!
#    print X_data_ave[1].shape
    logl,theta_list,pimix_list = EMl.EM(X_data_ave[1], K = 20, delta = 0.1, T = 100)
    
    gl.plot([],np.array(logl).flatten()[1:])
#    Ks_params[0][1]
##################### Plot the average probability of every cluster for train and test ###########
## TODO: Separate by label to see if the clusters behave differently to the labels
plot_clusters_info = 0
if (plot_clusters_info):
    # If we want to plot things about the clusters
#    subset = [X_train[i] for i in range (5) ]  # To plot a subset if wanted
#    dp.plot_Clusters_time(X_train,Ks_params)
#    dp.plot_Clusters_time(X_test,Ks_params)
    X_data_ave = dp.get_average_from_train(Nclasses, X_train_notNormalized, y_train, 
                                           normalize = True, partitions = 1)

    dp.plot_Clusters_time([X_data_ave[class_i]],[Ks_params])
#    dp.plot_Clusters_HMM_time([X_data_ave[class_i]],[Ks_params_HMM])
#    dp.plot_Clusters_time([X_data_ave[1]],Ks_params)
systems = 1
if (systems):
    ## Classify using just the likelihood of a trial for the 2 sets of clusters
    #Likelihoods_tr, Likelihoods_tst = sm.classify_with_Likelihood_EM(X_train, X_test, y_train, y_test, Ks_params)

    ## TODO: some of the values are Nan
    # Get the likelihood of every cluster and normalize data to have a normal ML probem
    Xtrain, Xtest, Ytrain, Ytest = sm.get_normalized_ll_byCluster_EM(X_train, X_test, y_train, y_test, Ks_params)
    
    Xtrain = np.concatenate((Xtrain,np.exp(Xtrain)), axis = 1)
    Xtest = np.concatenate((Xtest,np.exp(Xtest)), axis = 1)
    
    lr =  sm.get_LogReg(Xtrain, Xtest, Ytrain, Ytest)
    lda =  sm.get_LDA(Xtrain, Xtest, Ytrain, Ytest)
    qda =  sm.get_QDA(Xtrain, Xtest, Ytrain, Ytest)
    
    gnb = sm.get_GNB(Xtrain, Xtest, Ytrain, Ytest)
    gknn = sm.get_KNN(Xtrain, Xtest, Ytrain, Ytest)
    
#    gtree = sm.get_TreeCl(Xtrain, Xtest, Ytrain, Ytest)
#    rf = sm.get_RF(Xtrain, Xtest, Ytrain, Ytest, gtree)
#    ert = sm.get_ERT(Xtrain, Xtest, Ytrain, Ytest, gtree)
    
    glsvm =  sm.get_LSVM(Xtrain, Xtest, Ytrain, Ytest)
    gsvm_rf =  sm.get_SVM_rf(Xtrain, Xtest, Ytrain, Ytest)
####################################################### 
######################### HMM ########################### 
####################################################### 
HMM_flag = 0
if (HMM_flag):
    Nit = 10; I = 3; R = 100
    
    Ks_params_HMM = None
    X_data_ave = dp.get_average_from_train(Nclasses, X_train_notNormalized, y_train, 
                                           normalize = True, partitions = 1)
    Is_params = sm.get_clusters_labels_HMM(Nclasses, X_data_ave, range(Nclasses), Ks_params = Ks_params_HMM,
                            Ninit =Nit, I  =  I, R  = R, verbose = 1)

#    Is_params = sm.get_clusters_labels_HMM(Nclasses, X_train, y_train, Ks_params = Ks_params_HMM,
#                            Nit =Nit, I  =  I, R  = R, verbose = 1)
#    
    Likelihoods_tr, Likelihoods_tst = sm.classify_with_Likelihood_HMM(X_train, X_test, y_train, y_test, Is_params)

CV_HMM = 0
if (CV_HMM):
    print "Doing some nasty HMMs"
    # We compute the EM for both clusters for the classes.
    # We do it for different kluster sizes K. We train with the means of train and we validate with the means of test.
    Klusters = [1,2,3,4,5,6,7,8,9]
    ll_train, ll_test, All_Ks_params = sm.perfrom_CV_HMM_classes(Nclasses, X_data_trials, X_data_labels, 
                          Klusters = Klusters, Ninit = 20, R = 100, nfolds = 2)
    Klusters = [1,2,3,4,5,6,7,8,9]
    culo = pkl.load_pickle("./OnePerson1FoldHMM.pkl")
    ll_train = culo[0]
    ll_test = culo[1]
    
    for ic in range(Nclasses):
        gl.plot(Klusters, np.array([ll_train[ic], ll_test[ic]]).T, legend = ["tr", "Val"], labels = ["HMM class = " + str(ic),"States","loglike"])


plot_precomputed_likelihoods_CV = 0
if (plot_precomputed_likelihoods_CV):
    cosasX1 = pkl.load_pickle("./OnePerson1FoldEM.pkl")
    cosasX2 = pkl.load_pickle("./OnePerson1FoldHMM.pkl")
    class_i = 0
    
    gl.plot(range(1,15),cosasX1[0][class_i], 
            legend = ["Train EM"], 
    labels = ["Validation of Number of clusters with LL","Number of clusters (K)","LL"], 
    lw = 4,
            fontsize = 25,   # The font for the labels in the title
            fontsizeL = 30,  # The font for the labels in the legeng
            fontsizeA = 20)
    
    gl.plot(range(1,15),cosasX1[1][class_i], nf = 0,
            legend = ["Validation EM"], 
    lw = 4,
            fontsize = 25,   # The font for the labels in the title
            fontsizeL = 30,  # The font for the labels in the legeng
            fontsizeA = 20)
    
    
    gl.plot(range(1,10),cosasX2[0][class_i], nf = 0,
            legend = ["Train HMM"], 
    lw = 4,
            fontsize = 25,   # The font for the labels in the title
            fontsizeL = 30,  # The font for the labels in the legeng
            fontsizeA = 20)
            
    gl.plot(range(1,10),cosasX2[1][class_i], nf = 0,
            legend = ["Validation HMM"], 
    lw = 4,
            fontsize = 25,   # The font for the labels in the title
            fontsizeL = 30,  # The font for the labels in the legeng
            fontsizeA = 20)
#            
