import numpy as np

def remove_baseline(Xdata):
    """Remove the baseline from the whole trial"""
    for i in range(len(Xdata)):
        Xbaseline = np.mean(Xdata[i][0:50],axis=0)
        Xdata[i] = Xdata[i]-Xbaseline #broadcasting the mean baseliene vector
    return Xdata

def cut_beginning(Xdata):
    """remove the first 50 points of the trials"""
    for i in range(len(Xdata)):
        Xdata[i] = Xdata[i][50:,:]
    return Xdata
