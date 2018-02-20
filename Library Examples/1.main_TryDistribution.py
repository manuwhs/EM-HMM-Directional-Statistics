# Change main directory to the main folder and import folders
import os
os.chdir("../")
import import_folders
# Classical Libraries
import matplotlib.pyplot as plt
import numpy as np
# Own graphical library

# Official libraries

from graph_lib import gl

import Watson_distribution as Wad
#import Watson_samplingMartin as Was
import Watson_estimators as Wae
import general_func as gf

import sampler_lib as sl

plt.close("all")

########################################################
""" With 2D circle """
########################################################

## Data generation parameters
Ndim = 2  # Number of dimensions of our generated data
Nsa = 1000  # Number of samples we will draw for the grid
Nsampling = 1000 # Number of samples we draw from the distribution

## Distribution parameters
kappa = 15   # "Variance" of the circular multivariate guassian
mu_angle = 0.3*np.pi  # Mean angle direction
mu = [np.cos(mu_angle), np.sin(mu_angle)]  # Mean direction. We transfor to cartesian

# Just one sample example
Xangle = 0.5 *np.pi
Xsample = [np.cos(Xangle), np.sin(Xangle)]
prob = np.exp(Wad.Watson_pdf_log(Xsample, [mu, kappa]))

# Draw 2D samples as transformation of the angle
Xalpha = np.linspace(0, 2*np.pi, Nsa)
Xdata = np.array([np.cos(Xalpha), np.sin(Xalpha)])
probs = []  # Vector with probabilities

for i in range(Nsa):
    probs.append(np.exp(Wad.Watson_pdf_log(Xdata[:,i],[mu,kappa]) ))

## Plotting
gl.set_subplots(1,3)

## Plot it in terms of (angle, prob)
gl.plot(Xalpha,np.array(probs), 
        legend = ["pdf k:%f, mu_angle: %f"%(kappa,mu_angle)], 
        labels = ["Watson Distribution", "Angle(rad)", "pdf"], nf = 1, na = 1)
        
# Plot it in polar coordinates
probs  = np.array(probs)
Basex =  np.cos(Xalpha)
Basey = np.sin(Xalpha)

X1 = (1 + probs) * np.cos(Xalpha)
X2 = (1 + probs) * np.sin(Xalpha)

ax2 = gl.plot(X1,X2, 
        legend = ["pdf k:%f, mu_angle: %f"%(kappa,mu_angle)], 
        labels = ["Watson Distribution", "Angle(rad)", "pdf"],
         nf = 1)   

ax2 = gl.plot(Basex,Basey, 
        legend = ["pdf k:%f, mu_angle: %f"%(kappa,mu_angle)], 
        labels = ["Circle"],
        ls = "--")   



ax2.axis("equal")
if (0):
    ## Generate samples
    RandWatson = Was.randWatson(Nsampling, mu, kappa)
    
    mu_est2, kappa_est2 = Wae.get_Watson_muKappa_ML(RandWatson)
    print "Real: ", mu, kappa
    print "Estimate: ", mu_est2, kappa_est2
    
    gl.scatter(RandWatson[:,0],RandWatson[:,1], 
            legend = ["pdf k:%f, mu_angle: %f"%(kappa,mu_angle)], 
            labels = ["Watson Distribution", "Angle(rad)", "pdf"],
             nf = 1, na = 1, alpha = 0.1)   
         
#new_samples = sl.InvTransSampGrid(probs, Xalpha, Nsa * 2)

if(0):
    ########################################################
    """ With 3D sphere """
    ########################################################
    
    # Data generation parameters
    Nsa = 100  # Number of samples we will draw
    Nsampling = 1000
    ## Distribution parameters
    kappa = 20  # "Variance" of the circular multivariate guassian
    #mu_angle = [0.3*np.pi, 0.8*np.pi]  # Mean angle direction
    mu_angle = [0.5*np.pi, 0.5*np.pi]  # Mean angle direction
    mu = [np.sin(mu_angle[0])*np.cos(mu_angle[1]), np.sin(mu_angle[0])*np.sin(mu_angle[1]) , np.cos(mu_angle[0])]  # Mean direction. We transfor to cartesian
    #mu = [0,0,1]
    mu = np.array([1,-1,-1])
    
    mu = np.array(mu)
    mu = mu / np.sqrt(np.sum(mu * mu))
    
    
    # Draw 2D samples as transformation of the angle
    Xthetta = np.linspace(0, 2*np.pi, Nsa) # Grid plotting
    Xfi = np.linspace(0, 2*np.pi, Nsa)
    probs = []  # Vector with probabilities
    for i in range(Nsa):
        probs.append([])
        for j in range(Nsa):
            XdataSample = [np.sin(Xthetta[i])*np.cos(Xfi[j]),
                           np.sin(Xthetta[i])*np.sin(Xfi[j]), 
                           np.cos(Xthetta[i])]
            probs[i].append(Wad.Watson_pdf(XdataSample,mu,kappa ))
    
    probs = np.array(probs).T
    
    ## Plotting
    gl.set_subplots(1,3)
    ## Plot it in terms of (angle, prob)
    gl.plot_3D(Xthetta,Xfi, np.array(probs))
    gl.plot_3D(Xthetta,Xfi, np.array(probs), project = "spher")
    
    
    mu = np.random.randn(5,1);
    mu = gf.normalize_module(mu.T).flatten()
    ## Generate samples
    RandWatson = Was.randWatson(Nsampling, mu, kappa)
    gl.scatter_3D(RandWatson[:,0],RandWatson[:,1], RandWatson[:,2])
    
    mu_est = Wae.get_MLmean(RandWatson)
    kappa_est = Wae.get_MLkappa(mu_est, RandWatson)
    
    
    mu_est2, kappa_est2 = Wae.get_Watson_muKappa_ML(RandWatson)
    print "Real: ", mu, kappa
    print "Estimate: ", mu_est2, kappa_est2
