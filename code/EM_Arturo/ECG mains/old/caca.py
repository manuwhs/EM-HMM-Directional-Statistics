# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:10:09 2017

@author: montoya
"""
import numpy as np
import distributions_func as difu
from scipy.special import hyp1f1
import HMM_libfunc2 as HMMlf

#print difu.get_cp_log(19,4)
#print np.log(difu.get_cp(19,4))

#print np.log(difu.Watson_pdf(mu,mu,200))
#print difu.Watson_pdf_log(mu,mu,200)


#print "----"
#print difu.kummer_log(0.5,4, 100)
#print "----"
#print np.log(hyp1f1(0.5,4, 100))


D = 10
kappa = 40
K = 3
kappas = [45, 34, 10]
Nsam = 100
mu = np.random.randn(D,K);
mu = difu.normalize_module(mu.T).T

data = np.random.randn(D,Nsam);
cp_logs = []
for k in range(K):
    cp_logs.append(difu.get_cp_log(D,kappas[k]))
cp_logs = np.array(cp_logs)

result = difu.Watson_K_pdf_log(data,mu,kappas,cp_logs)

import time

dvs = HMMlf.sum_logs([10,10,10])


if (0):
    print "Watson"
    t0 = time.time()
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    np.log(difu.Watson_pdf(data,mu,kappa))
    t1 = time.time()
    
    total1 = t1-t0
    
    t0 = time.time()
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    difu.Watson_pdf_log(data,mu,kappa)
    t1 = time.time()
    
    total2 = t1-t0
    
    print total1, total2


if (0):
    print "Kummer"
    t0 = time.time()
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
    difu.kummer_log(0.5,4, 100)
#    print difu.kummer_log(0.5,4, 100)
    t1 = time.time()
    
    total1 = t1-t0
    
    t0 = time.time()
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    np.log(hyp1f1(0.5,4, 100))
    
#    print np.log(hyp1f1(0.5,4, 100))
    
    t1 = time.time()
    
    total2 = t1-t0

    print total1, total2