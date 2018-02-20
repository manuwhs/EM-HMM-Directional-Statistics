import import_folders
import matplotlib.pyplot as plt
import numpy as np
from graph_lib import gl


plt.close("all")


mu = 0
kappa = 4.0 # mean and dispersion
s = np.random.vonmises(mu, kappa, 1000)
# Display the histogram of the samples, along with the probability density function:


#import matplotlib.pyplot as plt
#from scipy.special import i0
#plt.hist(s, 50, normed=True)
#x = np.linspace(-np.pi, np.pi, num=51)
#y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
#plt.plot(x, y, linewidth=2, color='r')
#plt.show()

from scipy.special import hyp1f1
from scipy.special import gamma

Ndim = 3
kappa = 4


mu = [1*np.pi]
kappa = 4

alpha = 0.3*np.pi
prob = Watson_pdf(alpha, mu, kappa)
cp = get_cp(Ndim, kappa)


alphas = np.linspace(0, 1.9*np.pi, 1000)
probs = []
for alph in alphas:
    probs.append(Watson_pdf([np.cos(alph), np.sin(alpha)],[np.cos(mu), np.sin(mu)], kappa ))

gl.plot(alphas,probs)



if(0):
    
    Nsamples = 100
    Ndim = 2
    tol = 0.000000001
    Xdata = np.random.randn(Nsamples,Ndim) + 2
    Module = np.sqrt(np.sum(np.power(Xdata,2),1))
    Module = Module.reshape(Nsamples,1)
    # Check that the modulus is not 0
    Xdata = Xdata[np.where(Module > tol)[0],:]
    Xdata = np.divide(Xdata,Module)
    
    gl.scatter(Xdata[:,0], Xdata[:,1])
    
    
    Xdata = np.random.randn(Nsamples,Ndim) - 2
    Module = np.sqrt(np.sum(np.power(Xdata,2),1))
    Module = Module.reshape(Nsamples,1)
    # Check that the modulus is not 0
    Xdata = Xdata[np.where(Module > tol)[0],:]
    Xdata = np.divide(Xdata,Module)
    
    gl.scatter(Xdata[:,0], Xdata[:,1], nf = 0)
    
    ### POLLAS 
    Nsamples = 1000
    Ndim = 3
    tol = 0.000000001
    
    Xdata = np.random.randn(Nsamples,Ndim) + 2
    Module = np.sqrt(np.sum(np.power(Xdata,2),1))
    Module = Module.reshape(Nsamples,1)
    # Check that the modulus is not 0
    Xdata = Xdata[np.where(Module > tol)[0],:]
    Xdata = np.divide(Xdata,Module)
    
    gl.scatter_3D(Xdata[:,0], Xdata[:,1], Xdata[:,2])
    
    
    Nsamples = 100
    Xdata = np.random.randn(Nsamples,Ndim) - 0
    Module = np.sqrt(np.sum(np.power(Xdata,2),1))
    Module = Module.reshape(Nsamples,1)
    # Check that the modulus is not 0
    Xdata = Xdata[np.where(Module > tol)[0],:]
    Xdata = np.divide(Xdata,Module)
    
    gl.scatter_3D(Xdata[:,0], Xdata[:,1],Xdata[:,2], nf = 0)
    
