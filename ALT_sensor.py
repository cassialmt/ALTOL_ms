#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 00:42:23 2025

@author: manlow
"""
#import
import mne
from os.path import join
from os import chdir
from os import listdir
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
import pickle #for save_obj
from scipy.stats import linregress

#SVM imports
import timeit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import random
#% ===========================================================================
# FUNCTIONS
# =============================================================================
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f: #store in subject path /date/expt1/channel
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
#==============================================================================
#%% Set paths & conditions
#==============================================================================
#  compute path
path = '/home/manlow/analyses/attention_music/data/MEG/'
# subjects = [join('NatMEG_0005', '180411')]
fif_path = '/archive/20058_attention_music/MEG'
subjects_dir = '/home/manlow/analyses/attention_music/data/MRI/'
script_path = '/home/manlow/analyses/attention_music/scripts/python/'
fig_path ='/home/manlow/analyses/attention_music/figures/'
#==============================================================================
group_path=join(path,'Group')
subjects = ['NatMEG_0231','NatMEG_0110','NatMEG_0073','NatMEG_0213','NatMEG_0290','NatMEG_0177','NatMEG_0035','NatMEG_0245','NatMEG_0264','NatMEG_0063','NatMEG_0455','NatMEG_0216','NatMEG_0453','NatMEG_0457','NatMEG_0460','NatMEG_0461','NatMEG_0442','NatMEG_0462','NatMEG_0463','NatMEG_0464','NatMEG_0465','NatMEG_0467','NatMEG_0468','NatMEG_0469','NatMEG_0472','NatMEG_0476','NatMEG_0029','NatMEG_0005']
dates = ['171207','171208','171211','171212','171221','180115','180117','180122','180126','180131','180209','180214','180214','180216','180220','180221','180222','180222','180223','180223','180226','180227','180227','180228','180302','180306','180307','180411']
MSI = [72, 94, 80, 105, 69, 53, 121, 95, 132, 106, 73, 58, 127, 126, 63, 111, 40, 80, 82, 87, 47, 64, 92, 112, 55, 103, 99, 130]
perform= [0.71,0.61,0.21,0.86,0.71,0.393,0.96,0.57,1,0.96,0.54,0.32,0.86,1,0.68,0.96,0.39,0.46,0.46,0.93,0.93,0.39,0.61,1,0.39,0.79,0.86,1]
raw_filenames = ['dualvoice_tsss_mc.fif']

conditions =[
'o39' #BU onset
,'o43'
,'a39' #TD attend
,'a43']

#M4.1 contrasts for decoding
contrasts = []
contrasts.insert(0,['o39','o43']) #
contrasts.insert(1,['a39','a43']) #
# %% M2 Cleaning: ICA & Trial rejection
# =============================================================================\
# M2.1 ICA
for s in range(0,len(subjects)):
    raw = mne.io.read_raw_fif(join(fif_path,subjects[s],dates[s],raw_filenames[0])) 
    raw.load_data() 
    raw.pick_types(meg=True,stim=True, eog=True, ecg=True)
    raw.filter(h_freq=50, l_freq=1) #filter to remove drifts
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
    chdir(join(path,subjects[s],dates[s],'expt1B','channel'))
    raw_ica.save('raw_ica.fif') #save raw_ica
    ica.save('raw-ica.fif') #save ica

    #M3 Create epochs
    events = mne.find_events(raw_ica, stim_channel=['STI007'], min_duration=0.02, uint_cast=True)
    events7 = mne.merge_events(events, [5], 7, replace_events=True)

    events = mne.find_events(raw_ica, stim_channel=['STI008'], min_duration=0.02, uint_cast=True)
    events8 = mne.merge_events(events, [5], 8, replace_events=True)

    events = mne.find_events(raw_ica, stim_channel=['STI009'], min_duration=0.02, uint_cast=True)
    events9 = mne.merge_events(events, [5], 9, replace_events=True)

    events = mne.find_events(raw_ica, stim_channel=['STI010'], min_duration=0.02, uint_cast=True)
    events10 = mne.merge_events(events, [5], 10, replace_events=True)
    #define conditions
    events79_time = np.intersect1d(events7[:,0],events9[:,0]) #find timepoints for lowbot
    events79=np.zeros((len(events79_time),3),dtype=int);events79[:,0]=events79_time;events79[:,2]=int(79)

    events710_time = np.intersect1d(events7[:,0],events10[:,0]) #find timepoints for lowtop
    events710=np.zeros((len(events710_time),3),dtype=int);events710[:,0]=events710_time;events710[:,2]=int(710)

    events89_time = np.intersect1d(events8[:,0],events9[:,0]) #find timepoints for highbot
    events89=np.zeros((len(events89_time),3),dtype=int);events89[:,0]=events89_time;events89[:,2]=int(89)

    events810_time = np.intersect1d(events8[:,0],events10[:,0]) #find timepoints for hightop
    events810=np.zeros((len(events810_time),3),dtype=int);events810[:,0]=events810_time;events810[:,2]=int(810)

    events_oa=np.concatenate((events79,events89,events710,events810))
    events_onset=np.concatenate((events7,events8)) #BUA
    events_attend=np.concatenate((events9,events10)) #TDA
    
     #Find first two notes in every block (cue+short note)
    events6 = mne.find_events(raw_ica, stim_channel=['STI006'], min_duration=0.02, uint_cast=True)    
    firsttwo=[[e1,e2] for e1,e2 in zip(events6[:,0],events6[1::,0]) if e2-e1 > 1500 and e2-e1 < 1700] #len(firsttwo) should =28 blocks
    firsttwo_times=[item for sublist in firsttwo for item in sublist] 
    #remove cue+short note
    events_onset=[event for event in events_onset if event[0] not in firsttwo_times] #len(events_onset)
    events_attend=[event for event in events_attend if event[0] not in firsttwo_times]#len(events_attend)
    events_oa=[event for event in events_oa if event[0] not in firsttwo_times] #len(events_oa)  
    # [e2[0]-e1[0] for e1,e2 in zip(events,events[1::])] #check that first two times are removed

    tmin, tmax = -0.1, 1.01                  
    event_dict={'o39a39': 79,'o39a43': 710, 'o43a39':89, 'o43a43':810}
    epochs_oa = mne.Epochs(raw_ica, events_oa,picks='data',
                          event_id=event_dict,
                          tmin = tmin,tmax=tmax,
                          baseline=None, preload=True)
    epochs_onset = mne.Epochs(raw_ica, events=events_onset,picks='data',
                          event_id={'o39': 7, 'o43':8},
                          tmin = tmin,tmax=tmax,
                          baseline=None, preload=True
    epochs_attend = mne.Epochs(raw_ica, events=events_attend,picks='data',
                          event_id={'a39': 9, 'a43':10},
                          tmin = tmin,tmax=tmax,
                          baseline=None, preload=True)

    epochs=mne.concatenate_epochs([epochs_oa,epochs_onset,epochs_attend])       
    
    epochs.shift_time(-0.01)
    print(epochs)
    chdir(join(path,subjects[s],dates[s],'expt1B','channel'))
    epochs.save('epochs_nocue_ica.fif',overwrite=True) 
#%% D2 Compute #trials per condition per subject
trials=[]
for s in range(0,len(subjects)): #s=0
    epochs=mne.read_epochs(join(path,subjects[s],dates[s],'expt1B','channel','epochs2_nocue2_ica.fif'),preload=False);
    for con in conditions: #con=conditions[0]  
        trials.append([subjects[s],con,len(epochs[con])])
print(trials)  
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
#%% M6 Create epochsubPSDs & repeat kfolds
n=5#how many groups
kreps=100 #how many times to repeat split
seeds=range(kreps)
#fft parameters
fmin,fmax=1,45
tmin,tmax=0,1
nfft=1000
nseg=round((tmax-tmin)*1000) #The segments will be zero-padded if n_fft > n_per_seg.
window='hann'
for s in range(0,len(subjects)): #s=0   s=13  len(subjects)
    epochs=mne.read_epochs(join(path,subjects[s],dates[s],'expt1B','channel','epochs2_nocue2_ica.fif'))
    #condition loop
    epochsubPSDs=[]
    for c in conditions: #c=conditions[0]
        dat=epochs[c]
        epochsubPSD=[]
        for k in range(kreps):
            #randomly sort epochs into n groups
            ind=np.array((range(len(dat))));random.Random(seeds[k]).shuffle(ind)
            grps=np.array_split(ind,n)  
            epochs_split=[dat[g] for g in grps]
            epochs_splitavg=[ep.average() for ep in epochs_split]#average across epochs in each group
#------------welch part
            epochsplitPSDs=[a.compute_psd(method='welch',fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax,n_fft=nfft, window=window, n_overlap=0, n_per_seg=nseg) for a in epochs_splitavg] #fft                        
            #combine into epoch structure
            esp_datas=[esp.get_data() for esp in epochsplitPSDs]# b)power spectrum PSD             
            evnt=np.zeros((n,3),dtype=int);evnt[:,2]=dat.events[0,2];evnt[:,0]=range(n) #make dummy events
            epochsubPSD.append(mne.EpochsArray(data=np.stack((esp_datas)),info=dat.info,tmin=.001,event_id=dat.event_id,events=evnt))#  
        epochsubPSDs.append(epochsubPSD)        
    directory_path = join(path,subjects[s],dates[s],'expt1B','channel','epochs2'+cue+'_'+window+'_ica')   
    if not os.path.exists(directory_path):
        mkdir(directory_path)
    else:
        print(f"Directory '{directory_path}' already exists.")              
    chdir(directory_path)
    print('Saving '+subjects[s])
    save_obj(epochsubPSDs,'epochsubPSDs_n'+str(n)+'_k'+str(kreps)) #save in channel dir
freqs_save=epochsplitPSDs[0].freqs
#%% M4 SVM classification
#%% M4.2B Design classifier (improved version from M4.2A that does not use mne svm function)
kernel='linear'#kernel='linear' kernel='rbf'
scoring='roc_auc'; #scoring=[]
clf = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel, C=1)) 
kreps,n=100,5 # kreps and n defined in M6 
skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0) #each cv fold contains both classes
#%% M4.3B SVM for k>1  
picks='meg'
for s in range(0,len(subjects)): #s=0   s=13 len(subjects) s=22
    chdir(join(path,subjects[s],dates[s],'expt1B','channel','epochs2'+cue+'_'+window+'_ica'))
    data=load_obj('epochsubPSDs_n'+str(n)+'_k'+str(kreps)) #data=epochsubPSDs     
    meanscoreslist=[]
    for con in contrasts:#con=contrasts[0] select contrast
        kscores=[]
        for k in range(kreps): #k=0
            freqscores=[];
            for i in range(len(data)):#i=0 #loop over conditions
                if list(data[i][k].event_id.keys())==[con[0]]:    
                    X1=data[i][k].get_data(picks=picks)
                if list(data[i][k].event_id.keys())==[con[1]]:  
                    X2=data[i][k].get_data(picks=picks)
            for f in range(45):#f=42
                X = np.concatenate((X1, X2), axis=0)[:,:,f] #data dataxsensors
                y1 = ['labelA'] * X1.shape[0]
                y2 = ['labelB'] * X2.shape[0]
                y = y1 + y2 #labels                   
                scores = cross_val_score(clf, X, y, cv=skf,scoring=scoring,n_jobs=16)#
                freqscores.append(np.average(scores))
            kscores.append(freqscores)
                #average across kscores
        meanscoreslist.append(np.average(kscores,axis=0)) #cons x freqs
    chdir(join(path,subjects[s],dates[s],'expt1B','channel','SVM'))
    save_obj(meanscoreslist,'msl'+window+'_ica_'+picks+cue+'_epochs2') #bw2taper for 1s gives freq 2-45 Hz #name error:all meg mag& grad are used
    print('Saved SVM '+subjects[s])
#%% 4.3ppp Plot group butterfly
from scipy.stats import sem
filename='mslhann_ica_meg_nocue2_epochs2'
avgtype='mean'
alpha=0.3
#-------------------------------
freqs=range(1,46) #for 1s fft
fmin,fmax=freqs.index(4),freqs.index(45)
fm39,fm43=freqs.index(39),freqs.index(43)
meanscoreslist_grpavg=[]    
plt.figure(figsize=(10,6))
for i in range(len(contrasts)):
    msl=[] #for grp meanscores
    plt.subplot(1,2,i+1)
    for s in range(len(subjects)):
        if subjects[s] in (['NatMEG_0453','NatMEG_0468','NatMEG_0467']):
            continue
        chdir(join(path,subjects[s],dates[s],'expt1B','channel','SVM'))
        meanscoreslist=load_obj(filename)
        p=meanscoreslist[i]
        msl.append(meanscoreslist[i])
        plt.plot(freqs[fmin:fmax],np.transpose(p)[fmin:fmax],alpha=alpha)
    if avgtype=='mean':
        meanscoreslist_grpavg.append(np.mean(msl,axis=0)) #mean
    elif avgtype=='median':
        meanscoreslist_grpavg.append(np.median(msl,axis=0)) #median       
    mean_vals = meanscoreslist_grpavg[i][fmin:fmax]
    sem_vals = sem(msl, axis=0)[fmin:fmax]
    plt.plot(freqs[fmin:fmax], mean_vals, color='k')    
    plt.fill_between(freqs[fmin:fmax], mean_vals - sem_vals, mean_vals + sem_vals, color='#808080', alpha=1)
    plt.title(str(contrasts[i])+' '+str(meanscoreslist_grpavg[i][fm39])[:4]+' '+str(meanscoreslist_grpavg[i][fm43])[:4])
    ax=plt.gca()
    ax.set_ylim(0.4,.7)
    ax.axvline(39, color='r', linestyle='--')
    ax.axvline(43, color='r', linestyle='--')
    ax.axhline(0.5,color='k',linestyle='--')
subs=len(msl)
plt.suptitle(filename+' '+avgtype+' subjects='+str(subs))    
plt.show()
#%% 4.3pppEX Plot TDA group butterfly +  show excluded subjects
from scipy.stats import sem
filename='mslhann_ica_meg_nocue2_epochs2'
avgtype='mean'
alpha=0.3
#-------------------------------
freqs=range(1,46) #for 1s fft
fmin,fmax=freqs.index(4),freqs.index(45)
fm39,fm43=freqs.index(39),freqs.index(43)
meanscoreslist_grpavg=[]    
plt.figure(figsize=(10,6))
i=-1
msl=[] #for grp meanscores
plt.subplot(1,2,1) #plot all
for s in range(len(subjects)):
    if subjects[s] in (['NatMEG_0453','NatMEG_0468','NatMEG_0467']):
        continue 
    chdir(join(path,subjects[s],dates[s],'expt1B','channel','SVM'))
    meanscoreslist=load_obj(filename)
    p=meanscoreslist[i]
    msl.append(meanscoreslist[i])
    plt.plot(freqs[fmin:fmax],np.transpose(p)[fmin:fmax],alpha=alpha)
if avgtype=='mean':
    meanscoreslist_grpavg.append(np.mean(msl,axis=0)) #mean
elif avgtype=='median':
    meanscoreslist_grpavg.append(np.median(msl,axis=0)) #median       
mean_vals = meanscoreslist_grpavg[0][fmin:fmax]
sem_vals = sem(msl, axis=0)[fmin:fmax]
plt.plot(freqs[fmin:fmax], mean_vals, color='k')    
plt.fill_between(freqs[fmin:fmax], mean_vals - sem_vals, mean_vals + sem_vals, color='#808080', alpha=1)
ax=plt.gca()
ax.axvline(39, color='r', linestyle='--')
ax.axvline(43, color='r', linestyle='--')
ax.axhline(0.5,color='k',linestyle='--')
subs=len(msl)
plt.suptitle(filename+' '+avgtype+' subjects='+str(subs))    
plt.show()

msl=[] #for grp meanscores
plt.subplot(1,2,2) #plot without exclusions
for s in range(smin,smax):
    if subjects[s] in (['NatMEG_0467']):
        continue 
    chdir(join(path,subjects[s],dates[s],'expt1B','channel','SVM'))
    meanscoreslist=load_obj(filename)
    p=meanscoreslist[i]
    msl.append(meanscoreslist[i])
    if subjects[s] in (['NatMEG_0453','NatMEG_0468']):
        plt.plot(freqs[fmin:fmax],np.transpose(p)[fmin:fmax],alpha=1)
    else:
        plt.plot(freqs[fmin:fmax],np.transpose(p)[fmin:fmax],alpha=alpha)
if avgtype=='mean':
    meanscoreslist_grpavg.append(np.mean(msl,axis=0)) #mean
elif avgtype=='median':
    meanscoreslist_grpavg.append(np.median(msl,axis=0)) #median       
mean_vals = meanscoreslist_grpavg[1][fmin:fmax]
sem_vals = sem(msl, axis=0)[fmin:fmax]
plt.plot(freqs[fmin:fmax], mean_vals, color='k')    
plt.fill_between(freqs[fmin:fmax], mean_vals - sem_vals, mean_vals + sem_vals, color='#808080', alpha=1)
ax=plt.gca()
ax.axvline(39, color='r', linestyle='--')
ax.axvline(43, color='r', linestyle='--')
ax.axhline(0.5,color='k',linestyle='--')
subs=len(msl)
plt.suptitle(filename+' '+avgtype+' subjects='+str(subs))    
plt.show() 
#%% M5p AUC correlations & Normality test for mean3943 only
import scipy.stats as stats
filename='mslhann_ica_meg_nocue2_epochs2'

# M5.1) Compute mean(AUC39,43)
X_types=['MSI','perform']
groupscoretype='mean AUC at 39 & 43 Hz'

for co in range(-2,0):
    plt.figure(figsize=(10,10))
    groupscoreavg=[]
    for s in range(len(subjects)): #s=8 
        chdir(join(path,subjects[s],dates[s],'expt1B','channel','SVM'))
        meanscoreslist=load_obj(filename) #LH/RH
        score39=meanscoreslist[co][38] 
        score43=meanscoreslist[co][42]
        scoreavg=np.mean([score39,score43])
        groupscoreavg.append(scoreavg);
    scoretype=groupscoreavg         
    for xt,X_type in enumerate(X_types):
        plt.subplot(2,2,xt+1)
        if X_type=='MSI':  
            X=[msi for i,msi in enumerate(MSI)]
        elif X_type=='perform':   
            X=[msi for i,msi in enumerate(perform)]
        y=[sc for i,sc in enumerate(scoretype)]
        plt.scatter(X,y) #plot scatterplot
        plt.show()    
        m, b, r, p, se = linregress(X, y)
        r_squared = r**2    
        line_label = f'$r$ = {r:.2f}\np = {p:.2g}' #only show r (2dp) and p (2sf)
        plt.plot(X, m*np.array(X) + b, color='blue', label=line_label) #plot least squares line
        plt.legend()
        plt.title(groupscoretype+' vs '+X_type+' for '+contrasts[co][0]+' vs '+contrasts[co][1]) #LH/RH MSI/Perform
        #Normality test
        plt.subplot(2,2,xt+3)    
        p_value = stats.shapiro(y)[1] # Shapiro-Wilk test for normality
        stats.probplot(y, plot=plt) 
        plt.annotate(f'p-value: {p_value:.4f}', xy=(0.05, 0.9), xycoords='axes fraction')
        plt.suptitle(contrasts[co][0]+' vs '+contrasts[co][1]+' Q-Q Plot for Normality') #LH/RH        
        plt.tight_layout()
        X=[];y=[];