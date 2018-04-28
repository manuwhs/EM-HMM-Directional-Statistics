import os,sys
sys.path.append(os.path.abspath('../'))
os.chdir('../')
import import_folders
import EM_libfunc as EMlf
import EM_lib as EMl
import general_func as gf
import numpy as np

#TODO:
#   - Ceate the distribution... so it can be initialize the paramters
#   - pass to the object the ditribution which is gonna be a class
#   - the distribution class will have a init params method
#   - pues eso.. variable de devolver toda la lista o ultimo valor likelihood 
class EMobj:
    num_clusters = None
    alpha  = None
    max_iter = None
    pi = None
    theta = None
    dist = None
    
    def __init__(self,K=3, alpha=0.01, max_iter=30):
        self.num_clusters = K 
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self,Xdata,pi_init=None,theta_init=None):
        logl,theta_list,pimix_list = EMl.EM(Xdata, K = self.num_clusters,
                                            delta =self.alpha, T = self.max_iter,
                                            pi_init = pi_init,theta_init = theta_init)
        return logl, theta_list, pimix_list




