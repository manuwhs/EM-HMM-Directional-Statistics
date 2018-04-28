
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
load_dataset = 1
if (load_dataset):
    X_All_labels, label_classes = dp.load_one_person_dataset(dataset_folder = "./dataset/", 
                                                          filename = 'face_scrambling_spm8proc_sub07.mat')
    
    X_All_labels = [X_All_labels[0], X_All_labels[2]]
    label_classes = [label_classes[0], label_classes[2]]
    
    Nclasses = len(X_All_labels)
# X_All_labels[Nclass][Ntrials per class][Trial (Ntime x Ndim)]
################################################################
######## Loading in lists and preprocessing! ##################
###############################################################
# Change i0 by a random list.
creating_data = 1
if (creating_data):
#    channel_sel = [0, 11, 21, 28, 36, 50, 60]
#    channel_sel = [20,21,22]
    channel_sel = range(0,70)  # Subset of channels selected
    max_trials = 500 # Number of trials per class

    # Now we remove from each 70-dimensional time point the average.
    X_All_labels = dp.remove_timePoints_average(X_All_labels)
    

    ## Remove the average of the 70 channels to each sample
    # We obtain the average of th 70 channels like in a normal ML problem
    # No matter the class and then remove them from each la
#    X_All_labels = dp.remove_features_average(X_All_labels)

    # Then select the desired channels and then make modulus = 1
    X_data_trials, X_data_labels = dp.preprocess_data_set (
                                    X_All_labels, label_classes, 
                                    max_trials = max_trials, channel_sel= channel_sel)

    ################# Separate in train and validation ############    
    r_seed = np.abs(int(100 * np.random.randn()))                
    X_train, X_test, y_train, y_test = train_test_split(X_data_trials, X_data_labels, test_size=0.50, random_state = r_seed, stratify = X_data_labels)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
    
def get_meanMax_trials(trials, L = 3):
    Ntrial = len(trials)
#    maxes = np.zeros((Ntrial,1))
    maxes = []
    for trial_i in range (Ntrial):
        Nsamples, Ndim = trials[trial_i].shape
        smoothed = []
        for dim_i in range (Ndim):
            smoothed_dim = movingaverage(trials[trial_i][:,dim_i],L)
            smoothed.append(smoothed_dim)
        smoothed = np.array(smoothed).T
#        print smoothed.shape
        mean_max = np.max(smoothed,axis = 0)
        maxes.append(mean_max)
#        maxes[trial_i,0] = mean_max
    
    maxes = np.array(maxes)
    print maxes.shape
    return maxes
    
def get_meanMin_trials(trials, L = 5):
    Ntrial = len(trials)
    maxes = np.zeros((Ntrial,1))
    for trial_i in range (Ntrial):
        Nsamples, Ndim = trials[trial_i].shape
        smoothed = []
        for dim_i in range (Ndim):
            smoothed_dim = movingaverage(trials[trial_i][:,dim_i],L)
            smoothed.append(smoothed_dim)
        smoothed = np.array(smoothed).T
        mean_max = np.min(np.min(smoothed,axis = 1))
        maxes[trial_i,0] = mean_max
    
    return maxes
    
EM_flag = 1
if (EM_flag):
    Ninit = 50
    K  =  5
    
    All_maxes = []
    for i in range(Nclasses):
        X_train_class_i = [ X_train[j] for j in np.where(np.array(y_train) == i)[0]]
        
        All_maxes.append(get_meanMax_trials(X_train_class_i))
            
#%% Normalize data
Xtrain = get_meanMax_trials(X_train)
Xtest = get_meanMax_trials(X_test)

Xtrain_min = get_meanMin_trials(X_train)
Xtest_min = get_meanMin_trials(X_test)

#gl.scatter(Xtrain_min,Xtrain)
#gl.scatter(Xtest_min,Xtest, nf = 0)

Ytrain = y_train
Ytest = y_test

Ntrain,Ndim = Xtrain.shape
Ntest, Ndim = Xtest.shape
mx = np.mean(Xtrain,axis=0,dtype=np.float64)
stdx = np.std(Xtrain,axis=0,dtype=np.float64)

Xtrain = np.divide(Xtrain-np.tile(mx,[Ntrain,1]),np.tile(stdx,[Ntrain,1]))
Xtest = np.divide(Xtest-np.tile(mx,[Ntest,1]),np.tile(stdx,[Ntest,1]))

LR_cl = 1
if (LR_cl == 1):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = lr.score(Xtrain,Ytrain)
    scores[1] = lr.score(Xtest,Ytest)
    print('Logistic Regression, train: {0:.02f}% '.format(scores[0]*100))
    print('Logistic Regression, test: {0:.02f}% '.format(scores[1]*100))

LDA_cl = 1
if (LDA_cl == 1):
    from sklearn.lda import LDA
    lda = LDA()
    lda.fit(Xtrain,Ytrain)
    scores = np.empty((4))
    scores[0] = lda.score(Xtrain,Ytrain)
    scores[1] = lda.score(Xtest,Ytest)
    print('LDA, train: {0:.02f}% '.format(scores[0]*100))
    print('LDA, test: {0:.02f}% '.format(scores[1]*100))

#%% Tree Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import  make_scorer     # To make a scorer for the GridSearch.

Tree_cl = 0
if (Tree_cl == 1):
    param_grid = dict()
    param_grid.update({'max_features':[None,'balanced']})
    param_grid.update({'max_depth':np.arange(1,21)})
    param_grid.update({'min_samples_split':np.arange(2,11)})
    gtree = GridSearchCV(DecisionTreeClassifier(),param_grid,scoring='precision',cv=4,refit=True,n_jobs=-1)
    gtree.fit(Xtrain,Ytrain)
    scores = np.empty((6))
    scores[0] = gtree.score(Xtrain,Ytrain)
    scores[1] = gtree.score(Xtest,Ytest)
    print('Decision Tree, train: {0:.02f}% '.format(scores[0]*100))
    print('Decision Tree, test: {0:.02f}% '.format(scores[1]*100))
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=1000,max_features=gtree.best_estimator_.max_features,max_depth=gtree.best_estimator_.max_depth,min_samples_split=gtree.best_estimator_.min_samples_split,oob_score=True,n_jobs=-1)
    rf.fit(Xtrain,Ytrain)
    scores[2] = rf.score(Xtrain,Ytrain)
    scores[3] = rf.score(Xtest,Ytest)
    print('Random Forest, train: {0:.02f}% '.format(scores[2]*100))
    print('Random Forest, test: {0:.02f}% '.format(scores[3]*100))
    
    # Extremely Randomized Trees
    ert = ExtraTreesClassifier(n_estimators=1000,max_features=gtree.best_estimator_.max_features,max_depth=gtree.best_estimator_.max_depth,min_samples_split=gtree.best_estimator_.min_samples_split,n_jobs=-1)
    ert.fit(Xtrain,Ytrain)
    scores[4] = ert.score(Xtrain,Ytrain)
    scores[5] = ert.score(Xtest,Ytest)
    print('Extremely Randomized Trees, train: {0:.02f}% '.format(scores[4]*100))
    print('Extremely Randomized Trees, test: {0:.02f}% '.format(scores[5]*100))

#%% SVM Classifier
# Params C, kernel, degree, params of kernel
SVM_cl = 1
if (SVM_cl == 1):
    
    # Parameters for the validation
    C = np.logspace(-3,3,10)
    p = np.arange(2,5)
    gamma = np.array([0.125,0.25,0.5,1,2,4])/200
    
    # Create dictionaries with the Variables for the validation !
    # We create the dictinary for every TYPE of SVM we are gonna use.
    param_grid_linear = dict()
    param_grid_linear.update({'kernel':['linear']})
    param_grid_linear.update({'C':C})
    
    param_grid_pol = dict()
    param_grid_pol.update({'kernel':['poly']})
    param_grid_pol.update({'C':C})
    param_grid_pol.update({'degree':p})
    
    param_grid_rbf = dict()
    param_grid_rbf.update({'kernel':['rbf']})
    param_grid_rbf.update({'C':C})
    param_grid_rbf.update({'gamma':gamma})
    
    
    param = [{'kernel':'linear','C':C}]
    param_grid = [param_grid_linear,param_grid_pol,param_grid_rbf]
    
    # Validation is useful for validating a parameter, it uses a subset of the 
    # training set as "test" in order to know how good the generalization is.
    # The folds of "StratifiedKFold" are made by preserving the percentage of samples for each class.
    stkfold = StratifiedKFold(Ytrain, n_folds = 5)
    
    # The score function is the one we want to minimize or maximize given the label and the predicted.
    acc_scorer = make_scorer(accuracy_score)

    # GridSearchCV implements a CV over a variety of Parameter values !! 
    # In this case, over C fo the linear case, C and "degree" for the poly case
    # and C and "gamma" for the rbf case. 
    # The parameters we have to give it are:
    # 1-> Classifier Object: SVM, LR, RBF... or any other one with methods .fit and .predict
    # 2 -> Subset of parameters to validate. C 
    # 3 -> Type of validation: K-fold
    # 4 -> Scoring function. sklearn.metrics.accuracy_score

    gsvml = GridSearchCV(SVC(class_weight='balanced'),param_grid_linear, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvmp = GridSearchCV(SVC(class_weight='balanced'),param_grid_pol, scoring = acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    gsvmr = GridSearchCV(SVC(class_weight='balanced'),param_grid_rbf, scoring =acc_scorer,cv = stkfold, refit = True,n_jobs=-1)
    
    gsvml.fit(Xtrain,Ytrain)
    gsvmp.fit(Xtrain,Ytrain)
    gsvmr.fit(Xtrain,Ytrain)
    
    trainscores = [gsvml.score(Xtrain,Ytrain),gsvmp.score(Xtrain,Ytrain),gsvmr.score(Xtrain,Ytrain)]
    testscores = [gsvml.score(Xtest,Ytest),gsvmp.score(Xtest,Ytest),gsvmr.score(Xtest,Ytest)]
    maxtrain = np.amax(trainscores)
    maxtest = np.amax(testscores)
    print('SVM, train: {0:.02f}% '.format(maxtrain*100))
    print('SVM, test: {0:.02f}% '.format(maxtest*100))
    
####################################################### 
######################### HMM ########################### 
####################################################### 
HMM_flag = 0
if (HMM_flag):
    Nit = 1
    verbose = 1
    Is_params = []
    
    I = 2
    init_with_EM = 0  # Flag to init with the EM
    
    for k in range(Nclasses): 
        pi_init, B_init, A_init = None, None, None
        
        if (init_with_EM):
            pi_init, B_init, A_init = HMMlf.get_initial_HMM_params_from_EM(Ks_params[k])
            I = pi_init.size
            
        X_train_class_k = [ X_train[j] for j in np.where(np.array(y_train) == k)[0]]
        X_train_class_k = [X_data_ave[k]]
        
        logl,B_list,pi_list, A_list = \
            HMMl.run_several_HMM(data = X_train_class_k,I = I,delta = 0.01, R = 20
                     ,pi_init = pi_init, A_init = A_init, B_init = B_init, Ninit = Nit, verbose = verbose)
    
        Is_params.append([pi_list[-1],A_list[-1], B_list[-1]])

    Likelihoods = dp.get_likelihoods_HMM(X_train, Is_params)
#    print [y_train, np.argmax(Likelihoods, axis = 1)]
    print "Train Accuracy %f" %(gf.accuracy(y_train, np.argmax(Likelihoods, axis = 1)))

    Likelihoods = dp.get_likelihoods_HMM(X_test, Is_params)
#    print [y_test, np.argmax(Likelihoods, axis = 1)]
    print "Test Accuracy %f" %(gf.accuracy(y_test, np.argmax(Likelihoods, axis = 1)))
