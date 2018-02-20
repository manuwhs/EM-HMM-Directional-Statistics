import numpy as np
import pandas as pd
import IPython
import scipy.io as sio
import os

def main():
    #HOW to load fake data and mix 3 clusters
    path = '../data/test_data/'
    fake_datasets = load_fake_data(path)
    previsualize_datasets(fake_datasets)
    selected_clust = [0,5,11]
    clust_pies = [0.3,0.5,0.2]
    phi,X = mix_datasets(fake_datasets,selected_clust,clust_pies)
    #HOW TO LOAD EEG DATA, PREPROCESS AND CONCATENATE
    path_EEG = '../data/Henson_data/'
    subjects,timestamps = load_EEG_data(path_EEG)
    subjects = preprocessing(subjects,timestamps,average_trials= True,
                             cut_beginning= False)
    concatenated = concatenate_subjects(subjects)
    # Some plottings of subjects
    plotting = False
    if plotting == True:
        import matplotlib.pyplot as plt
        trial = subjects[0]['famous']
        for i in range(70): 
            plt.plot(timestamps[63:],trial[i,:])
        plt.grid(); plt.show()
    IPython.embed()

def load_fake_data(path):
    mat_files = [x for x in os.listdir(path) if x.find('.mat')!=-1]
    datasets_dic = {'name':[],'type':[],'kappa':[],'mu':[],'data':[]}
    for fi in mat_files:
        mat = sio.loadmat(path+fi)
        datasets_dic['name'].append(fi)
        datasets_dic['type'].append(mat['clust'][0][0][0][0])
        datasets_dic['kappa'].append(mat['clust'][0][0][1][0][0])
        datasets_dic['mu'].append(mat['clust'][0][0][2][0])
        datasets_dic['data'].append(mat['clust'][0][0][3])
    return datasets_dic

def previsualize_datasets(datasets):
    labels = ['name','type','kappa','mu']
    subset = {k: datasets.get(k) for k in labels}
    df = pd.DataFrame(subset)
    print df

def mix_datasets(datasets,selected_clust,clust_pies):
    """Concatenate the selected clusters with the weigths from clust_pies"""
    assert(len(selected_clust)==len(clust_pies))
    assert(np.sum(np.array(clust_pies))==1)
    clust0 =  datasets['data'][selected_clust[0]]
    kappa0 = datasets['kappa'][selected_clust[0]]
    mu0 = datasets['mu'][selected_clust[0]]
    type0 = datasets['type'][selected_clust[0]]

    cut_idx = int(clust0.shape[0]*clust_pies[0])
    X = clust0[0:cut_idx,:]
    phi = {'type':[type0], 'kappa':[kappa0], 'mu':[mu0],'pies':clust_pies}
    for i,idx in enumerate(selected_clust[1:]):
        phi['kappa'].append(datasets['kappa'][idx])
        phi['mu'].append(datasets['mu'][idx])
        phi['type'].append(datasets['type'][idx])
        clustN = datasets['data'][idx]
        cut_idx = int(clustN.shape[0]*clust_pies[i+1])
        X = np.concatenate([X,clustN[0:cut_idx,:]],axis=0)
    return phi,X    
        
def load_EEG_data(path):
    """load all the .mat files at path
    Input:
        path: path of .m files
    Output:
        subjects: return a list(n subjects). Each subjects contain a dcitionary
        for each condition. Each condition is a numpy array(trials*channels*samples)
    """
    mat_files = [x for x in os.listdir(path) if x.find('.mat')!=-1]
    subjects = []
    for fi in mat_files:
        mat = sio.loadmat(path+fi)
        famous_trials = []
        unfamiliar_trials = []
        scrambled_trials = []
        for i,label in enumerate(mat['labels']):
            if label==1:
                famous_trials.append(mat['data'][:,:,i])
            elif label == 2:
                unfamiliar_trials.append(mat['data'][:,:,i])
            else:
                scrambled_trials.append(mat['data'][:,:,i])
        subject =  {'famous':np.array(famous_trials),'unfamiliar':np.array(unfamiliar_trials),
                       'scrambled':np.array(scrambled_trials)}
        subjects.append(subject)
        print("Subject {} loaded..").format(fi)

    timestamps = [x for x in mat['time'][0]]
    return subjects,timestamps

def preprocessing(subjects,timestamps,average_trials = True, cut_beginning = True):
    for subject in subjects:
        for label in subject.keys():
            if cut_beginning == True:
                idx0 = [i for i,x in enumerate(timestamps) if x==0][0]
                subject[label] = subject[label][:,:,idx0:]
            if average_trials == True:
                subject[label]=subject[label].mean(axis=0)
    return subjects

def concatenate_subjects(subjects):
    concatenated = subjects.pop(0)
    famous = concatenated['famous']
    assert(len(famous.shape)==2)#check that trials have been averaged
    unfamiliar = concatenated['unfamiliar']
    scrambled = concatenated['scrambled']
    for subject in subjects:
        famous = np.concatenate([famous,subject['famous']],axis = 1)
        unfamiliar = np.concatenate([unfamiliar,subject['unfamiliar']],axis=1)
        scrambled = np.concatenate([scrambled,subject['unfamiliar']],axis=1)

    return {'famous':famous, 'unfamiliar':unfamiliar, 'scrambled':scrambled}

if __name__ == "__main__":
    main()
    
