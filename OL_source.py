#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 06:22:58 2025

@author: manlow
"""
#IMPORT===========================================================================
# =============================================================================
import mne
import os
from os.path import join, isfile
from os import chdir, listdir, mkdir
import numpy as np
import matplotlib.pyplot as plt
import pickle #for save_obj
# SVM import
import random
import timeit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import concurrent.futures
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
#Set paths
#==============================================================================
#  HPC path
path = '/proj/assr2023/data/MEG/'
fif_path = '/proj/assr2023/data/MEG_fif/MEG/'
subjects_dir = '/proj/assr2023/data/MRI/'
script_path = '/proj/assr2023/scripts/python/'

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

#%Categorizing labels
OrG = ['A14m_L-lh', 'A14m_R-rh', 'A12-47o_L-lh', 'A12-47o_R-rh', 'A11l_L-lh', 'A11l_R-rh', 'A11m_L-lh', 'A11m_R-rh', 'A13_L-lh', 'A13_R-rh', 'A12-47l_L-lh', 'A12-47l_R-rh']
STG=['A38m_L-lh', 'A38m_R-rh', 'A41-42_L-lh', 'A41-42_R-rh', 'TE1.0-TE1.2_L-lh', 'TE1.0-TE1.2_R-rh', 'A22c_L-lh', 'A22c_R-rh', 'A38l_L-lh', 'A38l_R-rh', 'A22r_L-lh', 'A22r_R-rh']
IPL=['A39c_L-lh', 'A39c_R-rh', 'A39rd_L-lh', 'A39rd_R-rh', 'A40rd_L-lh', 'A40rd_R-rh', 'A40c_L-lh', 'A40c_R-rh', 'A39rv_L-lh', 'A39rv_R-rh', 'A40rv_L-lh', 'A40rv_R-rh']

ROIs = [OrG, STG,IPL]
ROI_names =['OrG','STG','IPL']

#Define parameters
epochsfile='epochs_nocue2_ica'
output='cmb3943earlylate_nocue2'
kreps,n=100,5 # kreps and n
seeds=range(kreps)
#PSD parameters
window='boxcar'
fmin,fmax=30,45
tmin,tmax=0,1
fs=1000
nfft=round((tmax-tmin)*fs) #this sets the freq resolution
nseg=nfft#The segments will be zero-padded if n_fft > n_per_seg.
#------select freqs-------------
freqs=range(1,46) #for 1s fft
#-------------------------------
#Design classifier
kernel='linear'#kernel='linear' kernel='rbf'
scoring='roc_auc'; #scoring=[]
clf = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel, C=1)) 
skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0) #each cv fold contains both classes
#%% Run source svm
for s in range(len(subjects)):
    starttime = timeit.default_timer()
    #--skip if name exists---
    subject_path = join(path, subjects[s],dates[s],'expt1C')
    if isfile(join(subject_path,'source','SVM','n'+str(n)+'_k'+str(kreps),output+'.pkl')):
        print("cmb3943 exists for " + subjects[s])
        continue 
    print("starting s" +str(s)+", "+ subjects[s])    
    cmb3943=dict()
    subject_fif_path = join(fif_path, subjects[s],dates[s], raw_filenames[0])
    lab_dir=join(subjects_dir,subjects[s],'label','BN_Atlas')    
    ROI_labels=dict()
    for ROI,ROI_name in zip(ROIs,ROI_names):
        LH_labels =[mne.read_label(join(lab_dir,name+'.label')) for name in ROI if name.endswith('-lh')]
        RH_labels =[mne.read_label(join(lab_dir,name+'.label')) for name in ROI if name.endswith('-rh')]
        ROI_lh=LH_labels[0];ROI_rh=RH_labels[0];
        for L,R in zip(LH_labels[1::],RH_labels[1::]):
            ROI_lh+=L
            ROI_rh+=R
        ROI_labels[ROI_name+'_LH']=ROI_lh  
        ROI_labels[ROI_name+'_RH']=ROI_rh             
    epochs=mne.read_epochs(join(path,'HPC',subjects[s],'expt4_overlap',epochsfile+'.fif'))
    info = mne.io.read_info(subject_fif_path)
    fwd = mne.read_forward_solution(join(subject_path, 'model-fwd_dual.fif')) #load s13 NatMEG_0467 does not have MR
    # cov = mne.compute_covariance(epochs)
    cov = mne.read_cov(join(path, subjects[s],dates[s],'expt1', 'trials-cov.fif')) ## use same cov as expt2 as there are silent ISIs [306 x 306]  
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov) #compute inv      
    for lab,name in zip(ROI_labels,list(ROI_labels.keys())):
        print(name +" for " + subjects[s]+" s"+str(s)) 
        epochsubPSDs_early=dict() ;epochsubPSDs_late=dict()    
        for c in conditions: #c=conditions[0]
            dat=epochs[c]
            epochsubPSD_early=[];epochsubPSD_late=[];
            for k in range(kreps): #k=0
                print("k" +str(k)+", " +name+" for "+ subjects[s]+str(s))
                #randomly sort epochs into n groups
                ind=np.array((range(len(dat))));random.Random(seeds[k]).shuffle(ind)
                grps=np.array_split(ind,n)  
                #compute early
                epochs_split=[dat[g].crop(0,1) for g in grps]
                epochs_splitavg=[ep.average() for ep in epochs_split]#average across epochs in each group           
                stc_splitavg_all=[mne.minimum_norm.apply_inverse(esa, inv,pick_ori='vector',method='MNE') for esa in epochs_splitavg]#,label=ROI_labels[lab] restrict source estimation to label
                stc_splitavg=[a.in_label(ROI_labels[name]) for a in stc_splitavg_all] #select label vertices only
                psd_krep=[]
                for stc in stc_splitavg:              
                    stcfft=[mne.time_frequency.psd_array_welch(stc.data[:,i,:],sfreq=fs,fmin=39,fmax=43,n_fft=nfft,n_per_seg=nfft, window=window)[0] for i in range(3)] #welch
                    stcfftnorm=np.linalg.norm(stcfft,axis=0) #vertices x freqs
                    psd_krep.append(stcfftnorm) #len(psd_krep) =5
                epochsubPSD_early.append(psd_krep) #len(epochsubPSD)=kreps
                #compute late
                epochs_split=[dat[g].crop(1,2) for g in grps]
                epochs_splitavg=[ep.average() for ep in epochs_split]#average across epochs in each group           
                stc_splitavg_all=[mne.minimum_norm.apply_inverse(esa, inv,pick_ori='vector',method='MNE') for esa in epochs_splitavg]#,label=ROI_labels[lab] restrict source estimation to label
                stc_splitavg=[a.in_label(ROI_labels[name]) for a in stc_splitavg_all] #select label vertices only
                psd_krep=[]
                for stc in stc_splitavg:              
                    stcfft=[mne.time_frequency.psd_array_welch(stc.data[:,i,:],sfreq=fs,fmin=39,fmax=43,n_fft=nfft,n_per_seg=nfft, window=window)[0] for i in range(3)] #welch
                    stcfftnorm=np.linalg.norm(stcfft,axis=0) #vertices x freqs
                    psd_krep.append(stcfftnorm) #len(psd_krep) =5
                epochsubPSD_late.append(psd_krep) #len(epochsubPSD)=kreps
            epochsubPSDs_early[c]=epochsubPSD_early;epochsubPSDs_late[c]=epochsubPSD_late  
        data_early=epochsubPSDs_early;data_late=epochsubPSDs_late #len(data)=2;len(data[con])=100
        cmb3943[name]=np.zeros(len(contrasts))
        for icon,con in enumerate(contrasts):#con=contrasts[0] select contrast
            kscores=[]
            for k in range(kreps): #k=0          
                X1early=[np.concatenate((sd[:,0],sd[:,-1])) for sd in data_early[con[0]][k]] #compile 1st&last freq (39/43Hz)  to 2020features                        
                X2early=[np.concatenate((sd[:,0],sd[:,-1])) for sd in data_early[con[1]][k]]  
                X1late=[np.concatenate((sd[:,0],sd[:,-1])) for sd in data_late[con[0]][k]] #compile 1st&last freq (39/43Hz)                         
                X2late=[np.concatenate((sd[:,0],sd[:,-1])) for sd in data_late[con[1]][k]]                    
                X1=np.concatenate((X1early, X1late), axis=1) #n x 4040features
                X2=np.concatenate((X2early, X2late), axis=1)
                
                X = np.concatenate((X1, X2), axis=0) #data dataxfeatures
                y1 = ['labelA'] * np.array(X1).shape[0]
                y2 = ['labelB'] * np.array(X2).shape[0]
                y = y1 + y2 #labels                   
                scores = cross_val_score(clf, X, y, cv=skf,scoring=scoring,n_jobs=-1)#X.shape
                kscores.append(np.average(scores))                                    
            #average across kscores
            cmb3943[name][icon]=np.average(kscores,axis=0)                               
        data_early=[];data_late=[];           
    save_path=join(subject_path,'source','SVM','n'+str(n)+'_k'+str(kreps)) #edit dir
    print("Saving "+ subjects[s])
    save_obj(cmb3943,join(save_path,output)) #dict[ROIname][contrast]
    print(subjects[s]+' Processing time for decoding: {:.2f} sec'.format(timeit.default_timer() - starttime))
#%% M7.4 Collating data for plots
from scipy.stats import linregress

earlylate='earlylate'

filename='cmb3943'+earlylate+'_nocue2'
fileearly='cmb3943early_nocue2';filelate='cmb3943late_nocue2'
ROI_names=['OrG', 'STG', 'IPL']

names_LH=[roiname+'_LH' for roiname in ROI_names];names_RH=[roiname+'_RH' for roiname in ROI_names]
pltdat_LH = [[[], []] for _ in range(len(ROI_names))]#pltdat:[#ROISx1s/late1s]
pltdat_RH = [[[], []] for _ in range(len(ROI_names))]#

for s in range(0,len(subjects)):
    if subjects[s] in (['NatMEG_0467']):#missing data:
        continue
    subject_path = join(path, subjects[s],dates[s],'expt1C')
    chdir(join(subject_path,'source','SVM','n'+str(n)+'_k'+str(100)))     
    cmb3943_early=load_obj(fileearly);cmb3943_late=load_obj(filelate)
    for i in range(len(names_LH)):#LH
        pltdat_LH[i][0].append(cmb3943_early[names_LH[i]][0]) 
        pltdat_LH[i][1].append(cmb3943_late[names_LH[i]][0]) #ROI x earlylate
    for i in range(len(names_RH)): #RH
        pltdat_RH[i][0].append(cmb3943_early[names_RH[i]][0])   
        pltdat_RH[i][1].append(cmb3943_late[names_RH[i]][0])            
# chdir(join(path,'Group','expt1C','source','SVM'))
# save_obj(pltdat_LH,'pltdat_LH');save_obj(pltdat_RH,'pltdat_RH') #save
#%% M9 Source AUC cmb3943 correlations w MSI/performance  +QQ Normality plot
#run 7.4 first
# Scatterplot + correlations: Look for p<0.05 and plot individually
from scipy.stats import linregress
X_type='perform' #'perform' 'MSI'

if X_type=='MSI':    
    X=[msi for i,msi in enumerate(MSI) if subjects[i] not in ['NatMEG_0467']] 
elif X_type=='perform':   
    X=[msi for i,msi in enumerate(perform) if subjects[i] not in ['NatMEG_0467']]
for el,earlat in enumerate(['early','late']):
    corrpvalues_LH=[[] for _ in range(len(pltdat_LH))]#pltdat:[#ROIS] pltdat_LH[0][1] is empty due to legacy
    corrpvalues_RH=[[] for _ in range(len(pltdat_RH))]
    #LH
    fig, axes = plt.subplots(2,len(pltdat_LH), figsize=(15,10))
    for ri,attn in enumerate(pltdat_LH): #LH #ri=0;attn=pltdat_LH[ri]
        y=[sc for i,sc in enumerate(attn)][el] 
        m, b, r, p, se = linregress(X, y)
        corrpvalues_LH[ri]=p
        axes[0,ri].scatter(X,y)
        r_squared = r**2
        line_label = f'$r$ = {r:.2f}\np = {p:.2g}' #only show r (2dp) and p (2sf)
        axes[0,ri].plot(X, m*np.array(X) + b, color='blue', label=line_label) #use a-1 to plot TDA on toprow n BUA on botrow
        axes[0,ri].set_title(f'{names_LH[ri]}')
        axes[0,ri].set_xlabel(X_type) #'MSI' or 'Performance'
        axes[0,ri].set_ylabel('AUC')
        axes[0,ri].legend(loc='upper left')
        # Shapiro-Wilk test for normality
        p_value = stats.shapiro(y)[1] 
        stats.probplot(y, plot=axes[1,ri]) 
        axes[1,ri].set_title('Q-Q Normality Plot')
        axes[1,ri].set_xlabel('Theoretical Quantiles')
        axes[1,ri].set_ylabel('Ordered Values')
        axes[1,ri].annotate(f'p-value: {p_value:.4f}', xy=(0.05, 0.9), xycoords='axes fraction')    
    plt.suptitle('OVERLAP '+earlat+' LH AUC vs ' + X_type) #'MSI' or 'Performance'
    plt.tight_layout()
    #RH
    fig, axes = plt.subplots(2,len(pltdat_RH), figsize=(15,10))
    for ri,attn in enumerate(pltdat_RH): #RH
        y=[sc for i,sc in enumerate(attn)][el] 
        m, b, r, p, se = linregress(X, y)
        corrpvalues_RH[ri]=p
        axes[0,ri].scatter(X,y)
        r_squared = r**2
        line_label = f'$r$ = {r:.2f}\np = {p:.2g}' #only show r (2dp) and p (2sf)
        axes[0,ri].plot(X, m*np.array(X) + b, color='blue', label=line_label)
        axes[0,ri].set_title(f'{names_RH[ri]}')
        axes[0,ri].set_xlabel(X_type) #'MSI' or 'Performance'
        axes[0,ri].set_ylabel('AUC')
        axes[0,ri].legend(loc='upper left')
        # Shapiro-Wilk test for normality
        p_value = stats.shapiro(y)[1] 
        stats.probplot(y, plot=axes[1,ri]) 
        axes[1,ri].set_title('Q-Q Normality Plot')
        axes[1,ri].set_xlabel('Theoretical Quantiles')
        axes[1,ri].set_ylabel('Ordered Values')
        axes[1,ri].annotate(f'p-value: {p_value:.4f}', xy=(0.05, 0.9), xycoords='axes fraction') 
    plt.suptitle('OVERLAP '+earlat+' RH AUC vs ' + X_type) #'MSI' or 'Performance'
    plt.tight_layout()
#%% M10 (late-early) lateralization -Compile data RUN THIS FIRST
fileearly='cmb3943early_nocue2';filelate='cmb3943late_nocue2'
ROI_names=['OrG', 'STG', 'IPL'] 

names_LH=[roiname+'_LH' for roiname in ROI_names];names_RH=[roiname+'_RH' for roiname in ROI_names]
pltdat_LH = [[] for _ in range(len(ROI_names))]#pltdat:[#ROISx1s/late1s]
pltdat_RH = [[] for _ in range(len(ROI_names))]#

for s in range(0,len(subjects)):
    if subjects[s] in (['NatMEG_0467']):#missing data:
        continue
    subject_path = join(path, subjects[s],dates[s],'expt1C')
    chdir(join(subject_path,'source','SVM','n'+str(5)+'_k'+str(100))) 
    cmb3943_early=load_obj(fileearly);cmb3943_late=load_obj(filelate)
    for i in range(len(names_LH)):#LH
        earlylat=cmb3943_early[names_LH[i]][0];latelat=cmb3943_late[names_LH[i]][0]
        pltdat_LH[i].append((latelat-earlylat)/(earlylat+latelat)) 
    for i in range(len(names_RH)): #RH
        earlylat=cmb3943_early[names_RH[i]][0];latelat=cmb3943_late[names_RH[i]][0]
        pltdat_RH[i].append((latelat-earlylat)/(earlylat+latelat)) 
#%% M11 Source AUC cmb3943 earlyvslate lat w MSI/performance  +QQ Normality plot
#run M10 first
# Scatterplot + correlations: Look for p<0.05 and plot individually
from scipy.stats import linregress
func_list=[linregress]#pearsonr,kendalltau
corrpvalues_LH=dict();corrpvalues_RH=dict()

for X_type in ['MSI','perform']: #'perform' 'MSI'
    if X_type=='MSI':    
        X=[msi for i,msi in enumerate(MSI) if subjects[i] not in ['NatMEG_0467']] 
    elif X_type=='perform':   
        X=[msi for i,msi in enumerate(perform) if subjects[i] not in ['NatMEG_0467']] 
    
    for func in func_list:#func=func_list[0]
        corrpvalues_LH[X_type]=[[] for _ in range(len(pltdat_LH))]#pltdat:[#ROIS]xsubjects
        corrpvalues_RH[X_type]=[[] for _ in range(len(pltdat_RH))]
        #LH
        fig, axes = plt.subplots(2,len(pltdat_LH), figsize=(22,10))
        for ri,roi in enumerate(pltdat_LH): #LH #ri=0;roi=pltdat_LH[ri]
            y=[sc for sc in roi]
            if func==linregress:
                m, b, r, p, se = linregress(X, y)
            else:
                corr_coef, p = func(X, y)
            corrpvalues_LH[X_type][ri]=p
            axes[0,ri].scatter(X,y)
            r_squared = r**2
            line_label = f'$r$ = {r:.2f}\np = {p:.2g}'#only show r (2dp) and p (2sf)
            # line_label = f'y = {m:.3f}x + {b:.2f}\n$r$ = {r:.2f}\np = {p:.4f}'
            axes[0,ri].plot(X, m*np.array(X) + b, color='blue', label=line_label) #use a-1 to plot TDA on toprow n BUA on botrow
            axes[0,ri].set_title(f'{names_LH[ri]}')
            axes[0,ri].set_xlabel(X_type) #'MSI' or 'Performance'
            axes[0,ri].set_ylabel('lateralization index')
            axes[0,ri].legend(loc='upper left')
            # Shapiro-Wilk test for normality
            p_value = stats.shapiro(y)[1] 
            stats.probplot(y, plot=axes[1,ri]) 
            axes[1,ri].set_title('Q-Q Normality Plot')
            axes[1,ri].set_xlabel('Theoretical Quantiles')
            axes[1,ri].set_ylabel('Ordered Values')
            axes[1,ri].annotate(f'p-value: {p_value:.4f}', xy=(0.05, 0.9), xycoords='axes fraction')    
        plt.suptitle('OVERLAP LH AUC Late-Early lateralization vs ' + X_type+' '+func.__name__) #'MSI' or 'Performance'
        plt.tight_layout()
        #RH
        fig, axes = plt.subplots(2,len(pltdat_RH), figsize=(22,10))
        for ri,roi in enumerate(pltdat_RH): #RH
            y=[sc for sc in roi]#can add np.log here to normal correct
            if func==linregress:
                m, b, r, p, se = linregress(X, y)
            else:
                corr_coef, p = func(X, y)
            corrpvalues_RH[X_type][ri]=p
            axes[0,ri].scatter(X,y)
            r_squared = r**2
            # line_label = f'y = {m:.3f}x + {b:.2f}\n$r$ = {r:.2f}\np = {p:.4f}'
            line_label = f'$r$ = {r:.2f}\np = {p:.2g}' #only show r (2dp) and p (2sf)
            axes[0,ri].plot(X, m*np.array(X) + b, color='blue', label=line_label)
            axes[0,ri].set_title(f'{names_RH[ri]}')
            axes[0,ri].set_xlabel(X_type) #'MSI' or 'Performance'
            axes[0,ri].set_ylabel('lateralization index')
            axes[0,ri].legend(loc='upper left')
            # Shapiro-Wilk test for normality
            p_value = stats.shapiro(y)[1] 
            stats.probplot(y, plot=axes[1,ri]) 
            axes[1,ri].set_title('Q-Q Normality Plot')
            axes[1,ri].set_xlabel('Theoretical Quantiles')
            axes[1,ri].set_ylabel('Ordered Values')
            axes[1,ri].annotate(f'p-value: {p_value:.4f}', xy=(0.05, 0.9), xycoords='axes fraction') 
        plt.suptitle('OVERLAP RH AUC Late-Early lateralization vs ' + X_type+' '+func.__name__) #'MSI' or 'Performance'
        plt.tight_layout()