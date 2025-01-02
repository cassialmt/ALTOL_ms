#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 02:26:30 2025

@author: manlow
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#%% Expt 4/1C sensor analysis
#IMPORT===========================================================================
# =============================================================================
import mne
import os
from os.path import join, isfile
from os import chdir, listdir,mkdir
import numpy as np
import matplotlib.pyplot as plt
import pickle #for save_obj
import pandas as pd

# SVM import
import random
import timeit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import concurrent.futures

#Stats import
import scipy.stats as stats
from scipy.stats import linregress
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import permutation_test
#% ===========================================================================
# FUNCTIONS
# =============================================================================
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f: #store in subject path /date/expt1/channel
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def moving_average(data, window_size):
    window = np.ones(window_size) / window_size #Smooth the input data using a moving average filter
    return np.convolve(data, window, mode='same')
#==============================================================================
#Set paths
#==============================================================================
#  compute path -comment out b4 running on HPC
path = '/home/manlow/analyses/attention_music/data/MEG/'
fif_path = '/archive/20058_attention_music/MEG'
subjects_dir = '/home/manlow/analyses/attention_music/data/MRI/'
script_path = '/home/manlow/analyses/attention_music/scripts/python/'
fig_path ='/home/manlow/analyses/attention_music/figures/'
group_path=join(path,'Group')

subjects = ['NatMEG_0264','NatMEG_0063','NatMEG_0455','NatMEG_0216','NatMEG_0453','NatMEG_0457','NatMEG_0460','NatMEG_0461','NatMEG_0442','NatMEG_0462','NatMEG_0463','NatMEG_0464','NatMEG_0465','NatMEG_0467','NatMEG_0468','NatMEG_0469','NatMEG_0472','NatMEG_0476','NatMEG_0029','NatMEG_0005']
dates = ['180126','180131','180209','180214','180214','180216','180220','180221','180222','180222','180223','180223','180226','180227','180227','180228','180302','180306','180307','180411']
MSI = [132, 106, 73, 58, 127, 126, 63, 111, 40, 80, 82, 87, 47, 64, 92, 112, 55, 103, 99, 130]
perform=[1, 0.96, 0.68, 0.5, 1, 1, 0.82, 0.96, 0.43, 0.93, 0.57, 0.82, 0.82, 0.46, 0.61, 0.93, 0.39, 0.82, 0.79, 1]

raw_filenames = ['overlap_tsss_mc.fif']

conditions =[
'a39'
,'a43']

#contrasts for decoding
contrasts = []
contrasts.insert(0,['a39','a43']) #TDA
# %% M1 Cleaning: bpfilter + ICA
for s in range(len(subjects)): 
    raw = mne.io.read_raw_fif(join(fif_path,subjects[s],dates[s],raw_filenames[0])) 
    raw.load_data() 
    raw.pick_types(meg=True,stim=True, eog=True, ecg=True)
    raw.filter(h_freq=50, l_freq=1) #filter to remove drifts
#ica starts here        
    ica = mne.preprocessing.ICA(n_components=0.98, random_state=0)
    ica.fit(raw) ## apply ica on data
    ## find bad eog/heog/ecg indices automatically
    eog_indices, eog_scores = ica.find_bads_eog(raw,ch_name="EOG001")
    heog_indices, heog_scores = ica.find_bads_eog(raw,ch_name="EOG002")
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw,ch_name="ECG003")
    ica.exclude = eog_indices;ica.exclude += heog_indices;ica.exclude += ecg_indices;  
    # Remove artefact ICA components after checking
    raw_ica = raw.copy() # make a copy of the raw
    ica.apply(raw_ica) ## remove the components in "ica.exclude"
    #-----ica ends here---
    chdir(join(path,subjects[s],dates[s],'expt1C','channel'))
    raw.save('raw.fif', overwrite=True) #save raw
    raw_ica.save('raw_ica.fif') #save raw_ica
    ica.save('raw-ica.fif') #save ica
#%% M2C nocue2 Epochs & trial rejection ICA
for s in range(len(subjects)): #s=0
    raw_ica =mne.io.read_raw_fif(join(path,subjects[s],dates[s],'expt1C','channel','raw_ica.fif'))#raw_ica=raw
    events = mne.find_events(raw_ica, stim_channel=['STI007'], min_duration=0.02, uint_cast=True)    
    events7 = mne.merge_events(events, [5], 7, replace_events=True)
    
    events = mne.find_events(raw_ica, stim_channel=['STI008'], min_duration=0.02, uint_cast=True)
    events8 = mne.merge_events(events, [5], 8, replace_events=True)
    
    events = mne.find_events(raw_ica, stim_channel=['STI009'], min_duration=0.02, uint_cast=True)
    events9 = mne.merge_events(events, [5], 9, replace_events=True)
    
    events = mne.find_events(raw_ica, stim_channel=['STI010'], min_duration=0.02, uint_cast=True)
    events10 = mne.merge_events(events, [5], 10, replace_events=True)
    #define conditions
    A=np.array([e[0] for e in events7]);B=np.array([e[0] for e in events9]);#find timepoints for lowbot
    within_range = np.any(np.abs(A[:, None] - B) <= 100, axis=0) #use bottop times as they r more frequent than lowhigh times
    events79=np.zeros((len(B[within_range]),3),dtype=int);events79[:,0]=B[within_range];events79[:,2]=int(79)
    
    A=np.array([e[0] for e in events8]);B=np.array([e[0] for e in events10]);#find timepoints for hightop
    within_range = np.any(np.abs(A[:, None] - B) <= 100, axis=0)
    events810=np.zeros((len(B[within_range]),3),dtype=int);events810[:,0]=B[within_range];events810[:,2]=int(810) 
       
    events_cue=np.concatenate((events79,events810))
    
    #remove cue notes in every block (cue+short note)
    events6 = mne.find_events(raw_ica, stim_channel=['STI006'], min_duration=0.02, uint_cast=True)    
    firstone_times=[e1 for e1,e2 in zip(events6[:,0],events6[1::,0]) if e2-e1 > 1500 and e2-e1 < 1700]
    events_times=[e[0] for e in events_cue]
    
    A=np.array(events_times);B=np.array(firstone_times);
    within_range = np.any(np.abs(B[:, None] - A) <= 100, axis=0)
    events=[event for event in events_cue if event[0] not in A[within_range]]
    
    tmin, tmax = -0.1, 2.01
      
    event_dict={'a39':79,'a43': 810}
    epochs = mne.Epochs(raw_ica, events,picks='data',
              event_id=event_dict,
              tmin = tmin,tmax=tmax,
              baseline=None, preload=True)
    
    epochs.shift_time(-0.01)
    print(epochs)
    chdir(join(path,subjects[s],dates[s],'expt1C','channel'))
    epochs.save('epochs_nocue2_ica.fif',overwrite=True) 
#%% D3 Compute #ICA components removed
comps=np.zeros(len(subjects))
for s in range(len(subjects)): #s=0 s=23 s=4 s=13
    chdir(join(path,subjects[s],dates[s],'expt1B','channel'))
    ica=mne.preprocessing.read_ica('raw-ica.fif') #read ica from file if already present
    print(subjects[s]+': '+str(len(ica.exclude)))
    comps[s]=len(ica.exclude)
np.mean(comps) #check no zero values   
from scipy import stats
sem = stats.sem(comps)
#%% M5A Repeated kfolds epochsubPSDs 1h
window='boxcar'
#Set parameters
n=5#how many groups
kreps=100 #how many times to repeat split
seeds=range(kreps)
#fft parameters
fmin,fmax=1,45
fs=1000
nfft=fs #round((tmax-tmin)*fs) #this sets the freq resolution
nseg=nfft #The segments will be zero-padded if n_fft > n_per_seg.
starttime = timeit.default_timer()
for s in range(len(subjects)): 
    epochs=mne.read_epochs(join(path,subjects[s],dates[s],'expt1C','channel','epochs_nocue2_ica.fif'),preload=False)
    #condition loop
    epochsubPSDs=[];epochsubPSDs_early=[];epochsubPSDs_late=[]
    for c in conditions: #c=conditions[0]
        dat=epochs[c]
        epochsubPSD_early=[];epochsubPSD_late=[];
        for k in range(kreps):#k=0
            #randomly sort epochs into n groups
            ind=np.array((range(len(dat))));random.Random(seeds[k]).shuffle(ind)
            grps=np.array_split(ind,n)  
            epochs_split=[dat[g] for g in grps]
            epochs_splitavg=[ep.average() for ep in epochs_split]#average across epochs in each group
#------------welch part
            epochsplitPSDs_early=[a.compute_psd(method='welch',fmin=fmin,fmax=fmax, tmin=0, tmax=1,n_fft=nfft,n_per_seg=nseg, n_overlap=0, window=window) for a in epochs_splitavg]                           
            epochsplitPSDs_late=[a.compute_psd(method='welch',fmin=fmin,fmax=fmax, tmin=1, tmax=2,n_fft=nfft,n_per_seg=nseg, n_overlap=0, window=window) for a in epochs_splitavg]                            
            esp_datas_early=[esp.get_data() for esp in epochsplitPSDs_early]# b)power spectrum PSD             
            esp_datas_late=[esp.get_data() for esp in epochsplitPSDs_late]# b)power spectrum PSD             
            evnt=np.zeros((n,3),dtype=int);evnt[:,2]=dat.events[0,2];evnt[:,0]=range(n) #make dummy events
            epochsubPSD_early.append(mne.EpochsArray(data=np.stack((esp_datas_early)),info=dat.info,tmin=.001,event_id=dat.event_id,events=evnt))#times rep index not freq
            epochsubPSD_late.append(mne.EpochsArray(data=np.stack((esp_datas_late)),info=dat.info,tmin=.001,event_id=dat.event_id,events=evnt))#times rep index not freq
        epochsubPSDs_early.append(epochsubPSD_early)
        epochsubPSDs_late.append(epochsubPSD_late)
    dir_path=join(path,subjects[s],dates[s],'expt1C','channel','epochs_nocue2_ica')
    # mkdir(dir_path)
    chdir(dir_path)
    epochsubPSDs=[epochsubPSDs_early,epochsubPSDs_late]
    save_obj(epochsubPSDs,'epochsubPSDs_earlylate_'+window) #save
freqs_save=epochsplitPSDs_early[0].freqs
print('Processing time: {:.2f} sec'.format(timeit.default_timer() - starttime))
#%% M6 SVM classification
#%% M6.2 Design classifier 
kernel='linear'#kernel='linear' kernel='rbf'
scoring='roc_auc'; #scoring=[]
clf = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel, C=1)) 
#%% M6.3D SVM for k>1 ICA EARLY/LATE
earlylate='late'
window='boxcar'
freqs=list(freqs_save)
#-------------------------------
fmin=freqs.index(1)

kreps,n=100,5 # kreps and n defined in M6 
skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0) #each cv fold contains both classes
starttime = timeit.default_timer()
for s in range(0,len(subjects)): 
    chdir(join(path,subjects[s],dates[s],'expt1C','channel','epochs_nocue2_ica')) #
    data=load_obj('epochsubPSDs_earlylate_'+window)
    data_early=data[0];data_late=data[1]
    meanscoreslist=[]
    for con in contrasts:#con=contrasts[0] select contrast
        kscores=[]
        for k in range(kreps): #k=0
            freqscores=[];
            for i in range(len(data_early)):#i=0 #loop over conditions
                if list(data_early[i][k].event_id.keys())==[con[0]]:    
                    X1early=data_early[i][k].get_data(picks='meg') # n x ch x freqs
                if list(data_early[i][k].event_id.keys())==[con[1]]:  
                    X2early=data_early[i][k].get_data(picks='meg')                
            for j in range(len(data_late)):#j=0 #loop over conditions
                if list(data_late[j][k].event_id.keys())==[con[0]]:    
                    X1late=data_late[j][k].get_data(picks='meg') # n x ch x freqs      
                if list(data_late[j][k].event_id.keys())==[con[1]]:  
                    X2late=data_late[j][k].get_data(picks='meg')       
            for f in range(fmin,X1early.shape[-1]):#f=35
                if earlylate=='early':
                    X = np.concatenate((X1early, X2early), axis=0)[:,:,f] #data dataxsensors
                elif earlylate=='late':
                    X = np.concatenate((X1late, X2late), axis=0)[:,:,f] #data dataxsensors   
                y1 = ['labelA'] * X1.shape[0]
                y2 = ['labelB'] * X2.shape[0]
                y = y1 + y2 #labels 
                scores = cross_val_score(clf, X, y, cv=skf,scoring=scoring, n_jobs=8)#X.shape
                freqscores.append(np.average(scores))
            kscores.append(freqscores)
                #average across kscores
        meanscoreslist.append(np.average(kscores,axis=0)) #cons x freqs
    # os.mkdir(join(path,subjects[s],dates[s],'expt1C','channel','SVM'))
    chdir(join(path,subjects[s],dates[s],'expt1C','channel','SVM'))
    save_obj(meanscoreslist,'msl'+window+'_ica_'+earlylate+'1s_nocue2') 
print('Processing time for decoding: {:.2f} sec'.format(timeit.default_timer() - starttime))
#%% M6ppp Group butterplot for bestof2
from scipy.stats import sem
n,kreps =5,100
i=0
freqs=range(1,46) #for 1s fft
avgtype='mean' #mean/median
alpha=0.3
#-------------------------------
fmin,fmax=freqs.index(4),freqs.index(45)
fm39,fm43=freqs.index(39),freqs.index(43)
fileearly='mslboxcar_ica_early1s_nocue2';filelate='mslboxcar_ica_late1s_nocue2' #
plt.figure(figsize=(18,6))
grp_early=[];grp_late=[];grp=[] #for grp meanscores
for s in range(len(subjects)):
    chdir(join(path,subjects[s],dates[s],'expt1C','channel','SVM'))
    msl_early=load_obj(fileearly);msl_late=load_obj(filelate);msl=np.maximum(msl_early[i],msl_late[i])
    grp_early.append(msl_early[i]);grp_late.append(msl_late[i]);grp.append(msl)
    plt.subplot(1,3,1)
    plt.plot(freqs[fmin:fmax+1],np.transpose(msl_early[i])[fmin:fmax+1],alpha=alpha)
    plt.subplot(1,3,2)
    plt.plot(freqs[fmin:fmax+1],np.transpose(msl_late[i])[fmin:fmax+1],alpha=alpha)
    plt.subplot(1,3,3)
    plt.plot(freqs[fmin:fmax+1],np.transpose(msl)[fmin:fmax+1],alpha=alpha)
subs=len(msl)
#Plot early
plt.subplot(1,3,1) 
meanscoreslist_grpavg=[] 
if avgtype=='mean':
    meanscoreslist_grpavg.append(np.mean(grp_early,axis=0)) #compute mean/median
elif avgtype=='median':
    meanscoreslist_grpavg.append(np.median(grp_early,axis=0)) #compute mean/median 

mean_vals = meanscoreslist_grpavg[i][fmin:fmax+1]
sem_vals = sem(grp_early, axis=0)[fmin:fmax+1]
plt.plot(freqs[fmin:fmax+1], mean_vals, color='k')    
plt.fill_between(freqs[fmin:fmax+1], mean_vals - sem_vals, mean_vals + sem_vals, color='#808080', alpha=1)
ax=plt.gca()
# ax.axvline(1, color='r', linestyle='--')
ax.axvline(39, color='r', linestyle='--',linewidth=2)
ax.axvline(43, color='r', linestyle='--',linewidth=2)
ax.axhline(0.5,color='k',linestyle='--')
ax.set_ylim(0.4,0.7)
plt.title('early '+str(meanscoreslist_grpavg[0][fm39])[:4]+' '+str(meanscoreslist_grpavg[0][fm43])[:4]+' subjects='+str(len(grp_early)))
#Plot late
plt.subplot(1,3,2)
meanscoreslist_grpavg=[] 
if avgtype=='mean':
    meanscoreslist_grpavg.append(np.mean(grp_late,axis=0)) #compute mean/median
elif avgtype=='median':
    meanscoreslist_grpavg.append(np.median(grp_late,axis=0)) #compute mean/median  
mean_vals = meanscoreslist_grpavg[i][fmin:fmax+1]
sem_vals = sem(grp_late, axis=0)[fmin:fmax+1]
plt.plot(freqs[fmin:fmax+1], mean_vals, color='k')    
plt.fill_between(freqs[fmin:fmax+1], mean_vals - sem_vals, mean_vals + sem_vals, color='#808080', alpha=1)
ax=plt.gca()
# ax.axvline(1, color='r', linestyle='--')
ax.axvline(39, color='r', linestyle='--',linewidth=2)
ax.axvline(43, color='r', linestyle='--',linewidth=2)
ax.axhline(0.5,color='k',linestyle='--')
ax.set_ylim(0.4,0.7)
plt.title('late '+str(meanscoreslist_grpavg[0][fm39])[:4]+' '+str(meanscoreslist_grpavg[0][fm43])[:4]+' subjects='+str(len(grp_late)))
#Plot bestof2
plt.subplot(1,3,3)
meanscoreslist_grpavg=[] 
if avgtype=='mean':
    meanscoreslist_grpavg.append(np.mean(grp,axis=0)) #compute mean/median
elif avgtype=='median':
    meanscoreslist_grpavg.append(np.median(grp,axis=0)) #compute mean/median    
mean_vals = meanscoreslist_grpavg[i][fmin:fmax+1]
sem_vals = sem(grp, axis=0)[fmin:fmax+1]
plt.plot(freqs[fmin:fmax+1], mean_vals, color='k')    
plt.fill_between(freqs[fmin:fmax+1], mean_vals - sem_vals, mean_vals + sem_vals, color='#808080', alpha=1)
ax=plt.gca()
# ax.axvline(1, color='r', linestyle='--')
ax.axvline(39, color='r', linestyle='--',linewidth=2)
ax.axvline(43, color='r', linestyle='--',linewidth=2)
ax.axhline(0.5,color='k',linestyle='--')
ax.set_ylim(0.4,0.7)
plt.title('Best of earlylate '+str(meanscoreslist_grpavg[0][fm39])[:4]+' '+str(meanscoreslist_grpavg[0][fm43])[:4]+' subjects='+str(len(grp_early)))      
plt.suptitle(str(contrasts[i])+' '+avgtype)
plt.show()
#%%
#Set parameters
window='boxcar'
icaraw='ica'
tres=0.05 #set temporal resolution of fft window shift
#%% M2 Repeated kfolds epochsubPSDs ~10min/sub for tres=50ms
#Set parameters
tres=0.05 #set temporal resolution of fft window shift
n=5#how many groups
kreps=100 #how many times to repeat split
seeds=range(kreps)
#fft parameters
fs=1000
nfft=fs #round((tmax-tmin)*fs) #this sets the freq resolution
nseg=nfft #The segments will be zero-padded if n_fft > n_per_seg.
starttime = timeit.default_timer()    
for s in range(0,len(subjects)): #s=10 
    print("starting s" +str(s)+", "+ subjects[s])        
    meanscoreslist=[]  
    epochs=mne.read_epochs(join(path,subjects[s],dates[s],'expt1C','channel','epochs_nocue2_ica.fif'),preload=False)
    #condition loop
    epochsubPSDs=[];epochsubPSDs_early=[];
    for cont in contrasts: 
        conA=cont[0];conB=[cont[1]]
        datA=epochs[conA];datB=epochs[conB];
        kscores=[]
        for k in range(kreps):#k=0
            #randomly sort epochs into n groups
            ind=np.array((range(len(dat))));random.Random(seeds[k]).shuffle(ind)
            grps=np.array_split(ind,n)  
            epochs_split=[datA[g] for g in grps]
            epochsA_splitavg=[ep.average() for ep in epochs_split]#average across epochs in each group
            epochs_split=[datB[g] for g in grps]
            epochsB_splitavg=[ep.average() for ep in epochs_split]
#------------moving tmin,tmax loop-------------------------
            tscores=[[], []];
            for tmin in np.arange(0, 1+tres,tres):
                tmax=tmin+1
                # print(tmin, tmax)
                #welch part
                epochsplitPSDs=[a.compute_psd(method='welch',fmin=39,fmax=43, tmin=tmin, tmax=tmax,n_fft=nfft,n_per_seg=nseg, n_overlap=0, window=window) for a in epochsA_splitavg] #fft  , window='boxcar' gives better fm peaks                            
                esp_datasA=[esp.get_data() for esp in epochsplitPSDs]# b)power spectrum PSD             
                epochsplitPSDs=[a.compute_psd(method='welch',fmin=39,fmax=43, tmin=tmin, tmax=tmax,n_fft=nfft,n_per_seg=nseg, n_overlap=0, window=window) for a in epochsB_splitavg] #fft  , window='boxcar' gives better fm peaks                            
                esp_datasB=[esp.get_data() for esp in epochsplitPSDs]# b)power spectrum PSD  
                #----loop across 39 & 43 Hz
                for f in range(0,2): #f=0
                    X = np.concatenate((esp_datasA, esp_datasB), axis=0)[:,:,f] #data dataxsensors
                    y1 = ['labelA'] * len(esp_datasA)
                    y2 = ['labelB'] * len(esp_datasB)
                    y = y1 + y2 #labels 
                    scores = cross_val_score(clf, X, y, cv=skf,scoring=scoring, n_jobs=8)#X.shape
                    tscores[f].append(np.average(scores))
            kscores.append(tscores) #kreps x 39/43Hz x 1/tres
#-------------average across kscores----------------
    meanscoreslist.append(np.average(kscores,axis=0)) #39/43Hz x 1/tres
    chdir(join(path,subjects[s],dates[s],'expt1C','channel','SVM'))
    save_obj(meanscoreslist,'msl'+window+'_'+icaraw+'_'+str(tres)+'sdrift_nocue2')
print('Processing time: {:.2f} sec'.format(timeit.default_timer() - starttime))
#%% M2pB SS plots MSI_sorted; Plot 3943correlation r & p
func=kendalltau
window_size=0

MSI_subjects =[[x,i] for i,x in enumerate(MSI)]
MSI_subjects.sort() 
subjects_sort=[subjects[s] for m,s in MSI_subjects] #replace subjects with subjects_sort
dates_sort=[dates[s] for m,s in MSI_subjects] #replace dates with dates_sort
MSI_sort=np.sort(MSI) #replace MSI with MSI_sort
perform_sort=[perform[s] for m,s in MSI_subjects] 

filename='msl'+window+'_ica_'+str(tres)+'sdrift_nocue2'
plt.figure();figman=plt.get_current_fig_manager();figman.window.showMaximized() #maximize figure to window
grp_scores=[];corr_3943=[]
for s in range(len(subjects_sort)):
    plt.subplot(4,6,s+1)
    meanscoreslist=load_obj(join(path,subjects_sort[s],dates_sort[s],'expt1C','channel','SVM',filename)) 
    p=meanscoreslist[0]
    curve39,curve43=p[0],p[1]
    corr_coef, p_value = func(curve39,curve43)
    corr_3943.append([corr_coef,p_value,MSI_sort[s],perform_sort[s],subjects_sort[s]])
    grp_scores.append(meanscoreslist[0])
    plt.plot(np.arange(0,1+tres,tres),curve39,'green')
    plt.plot(np.arange(0,1+tres,tres),curve43,'purple')
    plt.xticks(np.arange(0,1.2,0.2))
    ax=plt.gca()
    ax.axhline(0.5,color='k',linestyle='--')
    if p_value<0.05:
        sig='*'
    else:
        sig=''
    plt.title('Subject_'+str(s)+', '+f'r = {corr_coef:.2g}'+sig,fontsize=14)
#plot grp_mean
grp_mean=np.mean(grp_scores,axis=0);grp_median=np.median(grp_scores,axis=0)
plt.subplot(4,6,23) 
plt.plot(np.arange(0,1+tres,tres),np.transpose(grp_mean))
ax=plt.gca()
ax.get_lines()[0].set_color('green')  # Change first line color to purple
ax.get_lines()[1].set_color('purple')
ax.axhline(0.5,color='k',linestyle='--')
plt.title('Group Mean',fontsize=14)
plt.subplot(4,6,24) #plot grp_median
plt.plot(np.arange(0,1+tres,tres),np.transpose(grp_median))
ax=plt.gca()
ax.get_lines()[0].set_color('green')  # Change first line color to purple
ax.get_lines()[1].set_color('purple')
ax.axhline(0.5,color='k',linestyle='--')
plt.title('Group Median',fontsize=14)
plt.legend(['39Hz','43Hz'])  
# plt.show()
plt.tight_layout()
#%% M3.4 Between-Group Early/Late attenders analysis using peaktimebest
from scipy.stats import ttest_ind

pthres=0.1
filename='msl'+window+'_ica_'+str(tres)+'sdrift_nocue2'

window_size=0 #window_size does not affect peak time
peaktimes=[]
for s in range(len(subjects)):
    meanscoreslist=load_obj(join(path,subjects[s],dates[s],'expt1C','channel','SVM',filename)) 
    p=meanscoreslist[0]
    curve39,curve43=p[0],p[1]
    mean3943=np.mean((curve39,curve43),axis=0)
    peak39,peak43,peak3943mean,peak3943best=np.max(curve39),np.max(curve43),np.max(mean3943),np.max(p)
    peakt3943best=np.arange(0,1+tres,tres)[np.argmax(p[np.argmax([peak39,peak43])])]
    peaktimes.append([peakt3943best,MSI[s],perform[s],subjects[s]])    
#Split groups based on peaktime [0 .5] or [.5 1]
grp_late = [item for item in peaktimes if abs(item[0]) >= .5] #[time, MSI, performance, subject] Late
grp_early = [item for item in peaktimes if abs(item[0]) < .5] #Early
print(len(grp_late),len(grp_early))  

#compare mean MSI & Per between groups
for x,X_type in enumerate(['MSI','perform']):       
    #ttest
    statistic, p_value = ttest_ind([item[x+1] for item in grp_late], [item[x+1] for item in grp_early])
    if p_value<=pthres:
        print('w='+str(window_size)+' '+X_type+" t-test statistic:", statistic.round(3)," p-value:", p_value.round(3))
    
    #permutation test: difference in mean/median late vs early
    def statistic(x, y):
        return np.mean(x) - np.mean(y)
    res = permutation_test(([item[x+1] for item in grp_late], [item[x+1] for item in grp_early]),statistic, n_resamples=10000,random_state=0)
    if res.pvalue<=pthres:
        print('w='+str(window_size)+' '+X_type+" Perm test p-value:", res.pvalue.round(3), "mean diff:",res.statistic.round(3))
        print(np.mean([item[x+1] for item in grp_late]).round(3),np.mean([item[x+1] for item in grp_early]).round(3))
#%%M3.4stats Early late attendees histrogram distribution
import seaborn as sns
df = pd.DataFrame({
    # 'Subject': [item[-1] for item in grp_early+grp_late],
    'Peaktimebest': [item[0] for item in grp_early+grp_late],
    'MSI': [item[1] for item in grp_early+grp_late],
    'Perform': [item[2] for item in grp_early+grp_late],
})

#plot Peaktimebest hist with 2 normal curves    
sns.displot(df, x='Peaktimebest',color='0.6',bins=10,binrange=(-0.05,1.05))#binwidth=0.1,binrange=(-0.05,1.05)
sns.kdeplot(df[0:10], x='Peaktimebest',bw_adjust=1.2)#
sns.kdeplot(df[11::], x='Peaktimebest')#,bw_adjust=.5
plt.ylabel('Participants')
plt.tight_layout()
#%% M3.4p Early vs Late attendees MSI/Per Boxplot 
colors = ['#009E73', '#D55E00']  # Green and Orange
categories = ['MSI', 'Performance']
sub_categories = ['early', 'late']
data=[[ [item[1] for item in grp_early],[item[1] for item in grp_late]],
    [[item[2] for item in grp_early],[item[2] for item in grp_late]]]
np.random.seed(9)
fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=False)
for i in range(len(categories)):
    axes[i].boxplot(data[i],showfliers=False, showcaps=False, whiskerprops={'visible': False},medianprops={'visible': False})
    for j in range(len(data[i])):
        y = data[i][j]
        x = np.random.normal(j+1, 0.04, size=len(y))  # add jitter to the x-axis so that points dont overlap
        axes[i].plot(x, y, 'o', markersize=4, markerfacecolor=colors[j])  
    means = [np.mean(data[i][0]), np.mean(data[i][1])]
    axes[i].plot([1, 2], means, marker='o', color='darkblue', linewidth=2) 
    axes[i].set_title(categories[i])
    axes[i].set_ylabel(categories[i])
    axes[i].set_xticklabels(sub_categories)
    axes[i].annotate(f'{np.mean(data[i][0]):.2f}', xy=(1, np.mean(data[i][0])),xytext=(10, 0), textcoords='offset points', ha='left', va='center', fontsize=12)
    axes[i].annotate(f'{np.mean(data[i][1]):.2f}', xy=(2, np.mean(data[i][1])),xytext=(10, 0), textcoords='offset points', ha='left', va='center', fontsize=12)
plt.suptitle('Early vs Late attendees separated by Best Peak time')
  #%% M5 Between-Group analysis based on curve39 and curve43 correlations
  pthres=0.1
  filename='msl'+window+'_ica_'+str(tres)+'sdrift_nocue2'

  func_list=[kendalltau]
  for window_size in range(1):
      corr_3943=dict()
      for func in func_list:#func=func_list[0]
          corr_3943[func.__name__]=[]
          for s in range(len(subjects)):
              meanscoreslist=load_obj(join(path,subjects[s],dates[s],'expt1C','channel','SVM',filename)) 
              p=meanscoreslist[0]
              curve39,curve43=p[0],p[1]
              corr_coef, p_value = func(curve39,curve43)
              corr_3943[func.__name__].append([corr_coef,p_value,MSI[s],perform[s],subjects[s]])
          #Split groups based on p_value[1]
          grp_above = [item for item in corr_3943[func.__name__] if item[1] > 0.05] #[corr_coeff, p_value, MSI,per,subject]
          grp_below = [item for item in corr_3943[func.__name__] if item[1] < 0.05]   
          #compare mean MSI & Per between groups
          for x,X_type in enumerate([' MSI',' perform']):              
              #permutation test
              def statistic(x, y):
                  return np.mean(x) - np.mean(y)
              res = permutation_test(([item[x+2] for item in grp_above], [item[x+2] for item in grp_below]),statistic, n_resamples=1000,random_state=0)
              if res.pvalue<=pthres:
                  print('w='+str(window_size)+' '+func.__name__+X_type+" Perm test p-value:", res.pvalue.round(3), "mean diff:",res.statistic.round(3))
                  print(np.mean([item[x+2] for item in grp_above]).round(3),np.mean([item[x+2] for item in grp_below]).round(3))
                  print(len(grp_above),len(grp_below))
#%% M5p Boxplot scatter corr vs uncorrelated
colors = ['#009E73', '#D55E00']  # Green and Orange
categories = ['MSI', 'Performance']
sub_categories = ['correlated', 'uncorrelated']
data=[[ [item[2] for item in grp_below],[item[2] for item in grp_above]],
    [[item[3] for item in grp_below],[item[3] for item in grp_above]]]
np.random.seed(9)
fig, axes = plt.subplots(1, 2, figsize=(7.5, 6), sharey=False)
for i in range(len(categories)):
    axes[i].boxplot(data[i],showfliers=False, showcaps=False, whiskerprops={'visible': False},medianprops={'visible': False})   
    for j in range(len(data[i])):
        y = data[i][j]
        x = np.random.normal(j+1, 0.05, size=len(y))  # add jitter to the x-axis so that points dont overlap
        axes[i].plot(x, y, 'o', markersize=4, markerfacecolor=colors[j])  
    means = [np.mean(data[i][0]), np.mean(data[i][1])]
    axes[i].plot([1, 2], means, marker='o', color='darkblue', linewidth=2) 
    axes[i].set_title(categories[i])
    axes[i].set_xticklabels(sub_categories, fontsize=16)
    axes[i].annotate(f'{np.mean(data[i][0]):.2f}', xy=(1, np.mean(data[i][0])),xytext=(10, 0), textcoords='offset points', ha='left', va='center', fontsize=12)
    axes[i].annotate(f'{np.mean(data[i][1]):.2f}', xy=(2, np.mean(data[i][1])),xytext=(10, 0), textcoords='offset points', ha='left', va='center', fontsize=12)
plt.suptitle('Correlated vs Uncorrelated attendees')