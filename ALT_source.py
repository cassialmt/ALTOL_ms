#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 01:59:18 2025

@author: manlow
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#%% Expt 3/1B source analysis for HPC
# =============================================================================
#import
import mne
from os.path import join, isfile, isdir
from os import chdir, listdir, mkdir
import numpy as np
import matplotlib.pyplot as plt
import pickle #for save_obj

# SVM imports
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
HPC_path=join(path,'HPC')
#==============================================================================
group_path=join(path,'Group')
subjects = ['NatMEG_0231','NatMEG_0110','NatMEG_0073','NatMEG_0213','NatMEG_0290','NatMEG_0177','NatMEG_0035','NatMEG_0245','NatMEG_0264','NatMEG_0063','NatMEG_0455','NatMEG_0216','NatMEG_0453','NatMEG_0457','NatMEG_0460','NatMEG_0461','NatMEG_0442','NatMEG_0462','NatMEG_0463','NatMEG_0464','NatMEG_0465','NatMEG_0467','NatMEG_0468','NatMEG_0469','NatMEG_0472','NatMEG_0476','NatMEG_0029','NatMEG_0005']
dates = ['171207','171208','171211','171212','171221','180115','180117','180122','180126','180131','180209','180214','180214','180216','180220','180221','180222','180222','180223','180223','180226','180227','180227','180228','180302','180306','180307','180411']
MSI = [72, 94, 80, 105, 69, 53, 121, 95, 132, 106, 73, 58, 127, 126, 63, 111, 40, 80, 82, 87, 47, 64, 92, 112, 55, 103, 99, 130]
perform = [0.71, 0.61, 0.21, 0.86, 0.71, 0.393, 0.96, 0.57, 1, 0.96, 0.54, 0.32, 0.86, 1, 0.68, 0.96, 0.39, 0.46, 0.46, 0.93, 0.93, 0.39, 0.61, 1, 0.39, 0.79, 0.86, 1]

raw_filenames = ['dualvoice_tsss_mc.fif']

conditions =[
'o39'
,'o43'
,'a39'
,'a43']

#M4.1 contrasts for decoding
contrasts = []
contrasts.insert(0,['o39','o43']) # BUA
contrasts.insert(1,['a39','a43']) # TDA
#%%Categorizing labels
OrG = ['A14m_L-lh', 'A14m_R-rh', 'A12-47o_L-lh', 'A12-47o_R-rh', 'A11l_L-lh', 'A11l_R-rh', 'A11m_L-lh', 'A11m_R-rh', 'A13_L-lh', 'A13_R-rh', 'A12-47l_L-lh', 'A12-47l_R-rh']
STG=['A38m_L-lh', 'A38m_R-rh', 'A41-42_L-lh', 'A41-42_R-rh', 'TE1.0-TE1.2_L-lh', 'TE1.0-TE1.2_R-rh', 'A22c_L-lh', 'A22c_R-rh', 'A38l_L-lh', 'A38l_R-rh', 'A22r_L-lh', 'A22r_R-rh']
IPL=['A39c_L-lh', 'A39c_R-rh', 'A39rd_L-lh', 'A39rd_R-rh', 'A40rd_L-lh', 'A40rd_R-rh', 'A40c_L-lh', 'A40c_R-rh', 'A39rv_L-lh', 'A39rv_R-rh', 'A40rv_L-lh', 'A40rv_R-rh']

ROIs = [OrG, STG,IPL]
ROI_names =['OrG','STG','IPL'] #3 x 2 ROIs 

#Define parameters
epochsfile='epochs2_nocue2_ica'
output='cmb3943_nocue2'
kreps,n=100,5 # kreps and n
seeds=range(kreps)
#PSD parameters
window='hann'
fmin,fmax=30,45
tmin,tmax=0,1
fs=1000
nfft=round((tmax-tmin)*fs) #this sets the freq resolution
nseg=nfft#The segments will be zero-padded if n_fft > n_per_seg.
#------select freqs-------------
freqs=range(fmin,fmax+1) #for 1s fft
#-------------------------------
#Design classifier
kernel='linear'#kernel='linear' kernel='rbf'
scoring='roc_auc'; #scoring=[]
clf = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel, C=1)) 
skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0) #each cv fold contains both classes
#%% Define function
for s in range(0,len(subjects)):
    starttime = timeit.default_timer()
    if subjects[s] =='NatMEG_0467':
        print("Skipping s21 NatMEG_0467 due to missing MR")
        continue
    #--skip if name file exists---
    subject_path = join(path, subjects[s],dates[s],'expt1B')
    if isfile(join(subject_path,'source','SVM','n'+str(n)+'_k'+str(kreps),output)+'.pkl'):
        print("cmb3943 exists for " + subjects[s])
        continue
    print("starting s" +str(s)+", "+ subjects[s])    
    cmb3943=dict()
    subject_fif_path = join(fif_path, subjects[s],dates[s], raw_filenames[0])
    lab_dir=join(subjects_dir,subjects[s],'label','BN_Atlas')    
    ROI_labels=dict()
    for ROI,ROI_name in zip(ROIs,ROI_names): #ROI=MFG
        LH_labels =[mne.read_label(join(lab_dir,name+'.label')) for name in ROI if name.endswith('-lh')]
        RH_labels =[mne.read_label(join(lab_dir,name+'.label')) for name in ROI if name.endswith('-rh')]
        ROI_lh=LH_labels[0];ROI_rh=RH_labels[0];
        for L,R in zip(LH_labels[1::],RH_labels[1::]):
            ROI_lh+=L
            ROI_rh+=R
        ROI_labels[ROI_name+'_LH']=ROI_lh  
        ROI_labels[ROI_name+'_RH']=ROI_rh             
    epochs=mne.read_epochs(join(path,'HPC',subjects[s],epochsfile+'.fif'))
    info = mne.io.read_info(subject_fif_path)
    fwd = mne.read_forward_solution(join(subject_path, 'model-fwd_alt.fif')) #load s13 NatMEG_0467 does not have MR
    cov = mne.read_cov(join(path, subjects[s],dates[s],'expt1', 'trials-cov.fif')) 
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov) #compute inv   
    for lab,name in zip(ROI_labels,list(ROI_labels.keys())):  
        print(name +" for " + subjects[s]+" s"+str(s)) 
        epochsubPSDs=dict()    
        for c in conditions: 
            dat=epochs[c]
            epochsubPSD=[]
            for k in range(kreps): #k=0
                print("k" +str(k)+", " +name+" for "+ subjects[s]+str(s))
                #randomly sort epochs into n groups
                ind=np.array((range(len(dat))));random.Random(seeds[k]).shuffle(ind)
                grps=np.array_split(ind,n)  
                epochs_split=[dat[g].crop(tmin,tmax) for g in grps]
                epochs_splitavg=[ep.average() for ep in epochs_split]#average across epochs in each group           
                stc_splitavg_all=[mne.minimum_norm.apply_inverse(esa, inv,pick_ori='vector',method='MNE') for esa in epochs_splitavg]#,label=ROI_labels[lab] restrict source estimation to label
                stc_splitavg=[a.in_label(ROI_labels[name]) for a in stc_splitavg_all] #select label vertices only
                psd_krep=[]
                for stc in stc_splitavg:              
                    stcfft=[mne.time_frequency.psd_array_welch(stc.data[:,i,:],sfreq=fs,fmin=39,fmax=43,n_fft=nfft,n_per_seg=nfft, window=window)[0] for i in range(3)] #welch
                    stcfftnorm=np.linalg.norm(stcfft,axis=0) #vertices x freqs
                    psd_krep.append(stcfftnorm) #len(psd_krep) =5
                epochsubPSD.append(psd_krep) #len(epochsubPSD)=kreps
            epochsubPSDs[c]=epochsubPSD 
        data=epochsubPSDs
        cmb3943[name]=np.zeros(len(contrasts))
        for icon,con in enumerate(contrasts):
            kscores=[]
            for k in range(kreps): #k=0
                X1=[np.concatenate((sd[:,0],sd[:,-1])) for sd in data[con[0]][k]] #compile 1st&last freq (39/43Hz)                         
                X2=[np.concatenate((sd[:,0],sd[:,-1])) for sd in data[con[1]][k]]        
                X = np.concatenate((X1, X2), axis=0) #data dataxfeatures
                y1 = ['labelA'] * np.array(X1).shape[0]
                y2 = ['labelB'] * np.array(X2).shape[0]
                y = y1 + y2 #labels                   
                scores = cross_val_score(clf, X, y, cv=skf,scoring=scoring,n_jobs=-1)#X.shape
                kscores.append(np.average(scores))                                    
            #average across kscores
            cmb3943[name][icon]=np.average(kscores,axis=0)
        data=[];     
    save_path=join(subject_path,'source','SVM','n'+str(n)+'_k'+str(kreps)) #edit dir
    print("Saving "+ subjects[s])
    save_obj(cmb3943,join(save_path,output))
    print(subjects[s]+' Processing time for decoding: {:.2f} sec'.format(timeit.default_timer() - starttime))
#%% P0 Brainmap plot #run this before opening spyder to plot in compute: export MESA_GL_VERSION_OVERRIDE=3.3 
subject='fsaverage-copy'

OrG = ['A14m_L-lh', 'A14m_R-rh', 'A12-47o_L-lh', 'A12-47o_R-rh', 'A11l_L-lh', 'A11l_R-rh', 'A11m_L-lh', 'A11m_R-rh', 'A13_L-lh', 'A13_R-rh', 'A12-47l_L-lh', 'A12-47l_R-rh']
STG=['A38m_L-lh', 'A38m_R-rh', 'A41-42_L-lh', 'A41-42_R-rh', 'TE1.0-TE1.2_L-lh', 'TE1.0-TE1.2_R-rh', 'A22c_L-lh', 'A22c_R-rh', 'A38l_L-lh', 'A38l_R-rh', 'A22r_L-lh', 'A22r_R-rh']
IPL=['A39c_L-lh', 'A39c_R-rh', 'A39rd_L-lh', 'A39rd_R-rh', 'A40rd_L-lh', 'A40rd_R-rh', 'A40c_L-lh', 'A40c_R-rh', 'A39rv_L-lh', 'A39rv_R-rh', 'A40rv_L-lh', 'A40rv_R-rh']
lab_dir=join(subjects_dir,subject,'label','BN_Atlas') 

ROIs = [OrG, STG,IPL]
ROI_names =['OrG','STG','IPL'] 
ROI_labels=dict()
for ROI,ROI_name in zip(ROIs,ROI_names): #ROI=MFG
    LH_labels =[mne.read_label(join(lab_dir,name+'.label')) for name in ROI if name.endswith('-lh')]
    RH_labels =[mne.read_label(join(lab_dir,name+'.label')) for name in ROI if name.endswith('-rh')]
    ROI_lh=LH_labels[0];ROI_rh=RH_labels[0];
    for L,R in zip(LH_labels[1::],RH_labels[1::]):
        ROI_lh+=L
        ROI_rh+=R
    ROI_labels[ROI_name+'_LH']=ROI_lh  
    ROI_labels[ROI_name+'_RH']=ROI_rh  
label_list=[ROI_labels[n+'_RH'] for n in ROI_names]
label_list_LH=[ROI_labels[n+'_LH'] for n in ROI_names]

brain = mne.viz.Brain('fsaverage-copy', subjects_dir=subjects_dir, background='white')
#Add labels
colors=['g','b','m']
for lab,c in zip(label_list,colors):    
    brain.add_label(lab,alpha=0.6,color=c,hemi=lab.hemi) #,borders='True'
# brain.show_view('lateral',row=0,col=1,hemi='lh')
chdir(join(fig_path,'manuscript4'))
# brain.save_image('brainlabels.tif')
#%% M7.4 Collating data for plots 
sub_categories = ['BUA','TDA']
ROI_names=['OrG', 'STG', 'IPL']#'MFG', 
n,kreps=5,100

names_LH=[roiname+'_LH' for roiname in ROI_names];names_RH=[roiname+'_RH' for roiname in ROI_names]
pltdat_LH = [[[] for _ in range(len(sub_categories))]  for _ in range(len(ROI_names))]#pltdat:[#ROISx6conts]
pltdat_RH = [[[] for _ in range(len(sub_categories))]  for _ in range(len(ROI_names))]#pltdat:[#ROISx6conts]

for s in range(0,len(subjects)):
    if subjects[s] in (['NatMEG_0467']):#noisy data 'NatMEG_0216' (<per),'NatMEG_0073' (<per), 'NatMEG_0467' (no MR)
    #noisy:'NatMEG_0453',
    # if subjects[s] in (['NatMEG_0467','NatMEG_0029','NatMEG_0463','NatMEG_0468']):#noisy data - doesn't really affect these plots:
        continue
    subject_path = join(path, subjects[s],dates[s],'expt1B')
    chdir(join(subject_path,'source','SVM','n'+str(n)+'_k'+str(100))) #make this dir for all subs
    cmb3943=load_obj('cmb3943_nocue2') #cmb3943[ROI][contrast]
    for i in range(len(ROI_names)):
        for co in range(len(sub_categories)):          
            pltdat_LH[i][co].append(cmb3943[names_LH[i]][co]) 
            pltdat_RH[i][co].append(cmb3943[names_RH[i]][co])     
# chdir(join(path,'Group','expt1B','source','SVM'))
# save_obj(pltdat_LH,'pltdat_LH');save_obj(pltdat_RH,'pltdat_RH') #save
#%% 7.4p Boxplot across subs (with ind datapoints)
# Jitter function
def add_jitter(points, jitter_strength=0.3):
    return points + np.random.uniform(-jitter_strength, jitter_strength, points.shape)

# Colorblind-friendly colors
colors = ['#009E73', '#D55E00']  # Green and Orange
# LH boxplot
fig, axes = plt.subplots(1, len(ROI_names), figsize=(7, 6), sharey=True)
for i in range(len(names_LH)):
    axes[i].boxplot(pltdat_LH[i], widths=0.2, showfliers=False, showcaps=False, whiskerprops={'visible': False},
                    medianprops={'visible': False})
    axes[i].set_title(names_LH[i] + ' ' + str(np.mean(pltdat_LH[i][0]))[0:5] + ' ' + str(np.mean(pltdat_LH[i][1]))[0:5])
    axes[i].set_xticklabels(sub_categories)
    axes[i].set_xlabel('Attention')
    # axes[i].set_ylabel('AUC')
    axes[i].axhline(y=0.5, color='k', linestyle='dotted')
    # Plot individual points with jitter and color
    for j in range(len(pltdat_LH[i][0])):
        j_x1 = add_jitter(np.array([1]))
        j_x2 = add_jitter(np.array([2]))
        axes[i].plot(j_x1, [pltdat_LH[i][0][j]], 'o', color=colors[0], alpha=0.5)
        axes[i].plot(j_x2, [pltdat_LH[i][1][j]], 'o', color=colors[1], alpha=0.5)
        # Connect individual points
        axes[i].plot([j_x1, j_x2], [pltdat_LH[i][0][j], pltdat_LH[i][1][j]], color='gray', alpha=0.5)
    # Connect means
    means = [np.mean(pltdat_LH[i][0]), np.mean(pltdat_LH[i][1])]
    axes[i].plot([1, 2], means, marker='o', color='darkblue', linewidth=2)
plt.suptitle('ALT LH AUC cmb3943 Boxplots')
plt.show()

# RH boxplot
fig, axes = plt.subplots(1, len(ROI_names), figsize=(7, 6), sharey=True)
for i in range(len(names_RH)):
    axes[i].boxplot(pltdat_RH[i], widths=0.3, showfliers=False, showcaps=False, whiskerprops={'visible': False},
                    medianprops={'visible': False})
    axes[i].set_title(names_RH[i] + ' ' + str(np.mean(pltdat_RH[i][0]))[0:5] + ' ' + str(np.mean(pltdat_RH[i][1]))[0:5])
    axes[i].set_xticklabels(sub_categories)
    axes[i].set_xlabel('Attention')
    # axes[i].set_ylabel('AUC')
    axes[i].axhline(y=0.5, color='k', linestyle='dotted')
    # Plot individual points with jitter and color
    for j in range(len(pltdat_RH[i][0])):
        j_x1 = add_jitter(np.array([1]))
        j_x2 = add_jitter(np.array([2]))
        axes[i].plot(j_x1, [pltdat_RH[i][0][j]], 'o', color=colors[0], alpha=0.5)
        axes[i].plot(j_x2, [pltdat_RH[i][1][j]], 'o', color=colors[1], alpha=0.5)
        # Connect individual points
        axes[i].plot([j_x1, j_x2], [pltdat_RH[i][0][j], pltdat_RH[i][1][j]], color='gray', alpha=0.5)
    # Connect means
    means = [np.mean(pltdat_RH[i][0]), np.mean(pltdat_RH[i][1])]
    axes[i].plot([1, 2], means, marker='o', color='darkblue', linewidth=2)
plt.suptitle('ALT RH AUC cmb3943 Boxplots')
#%% 7.4stats  Permutation test against chance=0.5 per ROI x Attend
from scipy.stats import permutation_test
def statistic(x, y):
    return np.mean(x) - np.mean(y)

pthres=0.1
pvalues=[] #[ROIname, meandiff, p]
#permutation test
for attn in range(2):
    for roi in range(3): #roi=0;attn=0
        dat=pltdat_LH[roi][attn]
        chancedat=np.full(len(dat),0.5)
        res = permutation_test((dat,chancedat),statistic, n_resamples=10000,random_state=0)
        pvalues.append([f"{sub_categories[attn]} {ROI_names[roi]}_LH",res.statistic,res.pvalue])
        #RH
        dat=pltdat_RH[roi][attn]
        chancedat=np.full(len(dat),0.5)
        res = permutation_test((dat,chancedat),statistic, n_resamples=10000,random_state=0)
        pvalues.append([f"{sub_categories[attn]} {ROI_names[roi]}_RH",res.statistic,res.pvalue])

#FDR correction: 
from statsmodels.stats.multitest import multipletests

rejected, corrected_pvalues,alphacsidak, alphacBonf = multipletests([item[-1] for item in pvalues], method='fdr_bh')
pvalues_fdr=[[item[0],item[1].round(3),item[2].round(5),pfdr.round(5)] for item,pfdr in zip(pvalues,corrected_pvalues)]
# Print corrected p-values and rejected hypotheses
[item for item in pvalues_fdr if item[-1] < 0.05]
print(len(corrected_pvalues))
#%% 7.4stats Permutation test btwen BUA & TDA
from scipy.stats import permutation_test
def statistic(x, y):
    return np.mean(x) - np.mean(y)

pthres=0.1

pvalues=[]
for r,roi in enumerate(ROI_names): #r=1; roi=ROI_names[r]
    BUA=pltdat_LH[r][0]
    TDA=pltdat_LH[r][1]
     #permutation test
    res = permutation_test((BUA,  TDA),statistic, n_resamples=10000,random_state=0)
    pvalues.append([roi+'_LH',res.statistic,res.pvalue])
    if res.pvalue<=pthres:
        print(roi+" LH Perm test p-value:", res.pvalue.round(3), "mean diff:",res.statistic.round(3))
    BUA=pltdat_RH[r][0]
    TDA=pltdat_RH[r][1]
     #permutation test
    res = permutation_test((BUA,  TDA),statistic, n_resamples=10000,random_state=0)
    pvalues.append([roi+'_RH',res.statistic,res.pvalue])
    if res.pvalue<=pthres:
        print(roi+" RH Perm test p-value:", res.pvalue.round(5), "mean diff:",res.statistic.round(3))
#FDR correction: 
from statsmodels.stats.multitest import multipletests
rejected, corrected_pvalues,alphacsidak, alphacBonf = multipletests([item[-1] for item in pvalues], method='fdr_bh')
pvalues_fdr=[[item[0],item[1].round(3),item[2].round(5),pfdr.round(5)] for item,pfdr in zip(pvalues,corrected_pvalues)]
# Print corrected p-values and rejected hypotheses
[item for item in pvalues_fdr if item[-1] < 0.05]
print(len(corrected_pvalues))
#%%M7.5 Test for normality for t-test incl correlations
#run 7.4 first
import scipy.stats as stats
alpha = 0.05 # Set significance level

for co in range(len(sub_categories)):   
    fig, axes = plt.subplots(1, 2*len(ROI_names), figsize=(25, 6), sharey=True)
    for i in range(len(ROI_names)): #i=0
        data =pltdat_LH[i][co] #np.log if needed
        p_value = stats.shapiro(data)[1] # Shapiro-Wilk test for normality
        stats.probplot(data, plot=axes[i]) 
        axes[i].set_title(f'{names_LH[i]}')
        axes[i].set_xlabel('Theoretical Quantiles')
        axes[i].set_ylabel('Ordered Values')
        axes[i].annotate(f'p-value: {p_value:.4f}', xy=(0.05, 0.9), xycoords='axes fraction')
        #RH
        data = pltdat_RH[i][co]#np.log if needed
        p_value = stats.shapiro(data)[1] # Shapiro-Wilk test for normality
        stats.probplot(data, plot=axes[i+len(ROI_names)]) 
        axes[i+len(ROI_names)].set_title(f'{names_RH[i]}')
        axes[i+len(ROI_names)].set_xlabel('Theoretical Quantiles')
        axes[i+len(ROI_names)].set_ylabel('Ordered Values')
        axes[i+len(ROI_names)].annotate(f'p-value: {p_value:.4f}', xy=(0.05, 0.9), xycoords='axes fraction')        
    plt.suptitle('AUC cmb3943 Q-Q Plot for Normality'+' '+sub_categories[co])
    plt.tight_layout()
#%% M9p Source AUC cmb3943 correlations w MSI/performance
#run 7.4 first
# Scatterplot + correlations: Look for p<0.05 and plot individually
X_type='perform' #'perform' 'MSI'
if X_type=='MSI':  
    X=[msi for i,msi in enumerate(MSI) if subjects[i] not in ['NatMEG_0467']]  #excludenatmeg0468, 0453 and i!=25; 
elif X_type=='perform':   
    X=[msi for i,msi in enumerate(perform) if subjects[i] not in ['NatMEG_0467']]  #

corrpvalues_LH=[[[] for _ in range(len(sub_categories))]  for _ in range(len(ROI_names))]#pltdat:[#ROISx6conts]
corrpvalues_RH=[[[] for _ in range(len(sub_categories))]  for _ in range(len(ROI_names))]#pltdat:[#ROISx6conts]

for co in range(len(sub_categories)):
    fig, axes = plt.subplots(2,len(ROI_names), figsize=(15,10))
    for ri in range(len(ROI_names)):
        #LH
        y=pltdat_LH[ri][co] #np.log if needed
        m, b, r, p, se = linregress(X, y)
        corrpvalues_LH[ri][co]=p
        axes[0,ri].scatter(X,y)
        r_squared = r**2
        line_label = f'$r$ = {r:.2f}\np = {p:.2g}' #only show r (2dp) and p (2sf)
        axes[0,ri].plot(X, m*np.array(X) + b, color='blue', label=line_label) #use ri to plot TDA on toprow n BUA on botrow
        axes[0,ri].set_title(f'{names_LH[ri]}')
        axes[0,ri].set_xlabel(X_type) #'MSI' or 'Performance'
        axes[0,ri].set_ylabel('AUC')
        axes[0,ri].legend(loc='upper left')
        #RH
        y=pltdat_RH[ri][co] #np.log if needed
        m, b, r, p, se = linregress(X, y)
        corrpvalues_RH[ri][co]=p
        axes[1,ri].scatter(X,y)
        r_squared = r**2
        line_label = f'$r$ = {r:.2f}\np = {p:.2g}' #only show r (2dp) and p (2sf)
        axes[1,ri].plot(X, m*np.array(X) + b, color='blue', label=line_label) #use ri to plot TDA on toprow n BUA on botrow
        axes[1,ri].set_title(f'{names_RH[ri]}')
        axes[1,ri].set_xlabel(X_type) #'MSI' or 'Performance'
        axes[1,ri].set_ylabel('AUC')
        axes[1,ri].legend(loc='upper left')          
    plt.suptitle('ALT AUC vs '+X_type+' correlation '+sub_categories[co]) #'MSI' or 'Performance'
    plt.tight_layout()
    # plt.savefig(join(fig_path,'dual_ALT','source','SVM','cmbcorrelations',sub_categories[co]+'_AUC_vs_'+X_type+'.png'))
