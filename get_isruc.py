import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
import math

#找到主要噪音位置，选择噪音长度添加，然后再合并成一个原长

def get_rms(records):   
    # return math.sqrt( sum([x ** 2 for x in records]) / len(records) ) 
    return np.sqrt( np.mean(records**2,axis=-1))
 
def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))# 原本psg是一个dict, 纬度是6000，应该是200Hz
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

def read_noise(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))# 原本psg是一个dict, 纬度是6000，应该是200Hz
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])

def contaminate(clean, artifact_ori, times = 1):

    # SNR_val_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_val_dB = 0
    SNR_val = 10 ** (0.1 * (SNR_val_dB))
    artifact = np.zeros((artifact_ori.shape[0],1,artifact_ori.shape[-1]))
    for c in range(artifact_ori.shape[1]):
       artifact +=  artifact_ori[:,c:c+1,:]
    # print(clean.shape)
    # print(artifact.shape)
    # exit()
    # combin eeg and noise for test set 
    noise_EEG = []
    noise_EEG_tmp=[]
    for i in range(times): # 1-10 snr ratio
        noise_eeg_val = []
        for j in range(clean.shape[0]): # batch-level   
            eeg = clean[j]
            noise = artifact[j]
            coe = get_rms(eeg) / (get_rms(noise) * SNR_val)
            
            noise = noise * coe.reshape(-1,1)
            
            neeg = noise + eeg
            
            noise_eeg_val.append(neeg)
        
        noise_EEG_tmp.extend(noise_eeg_val)
    noise_EEG_tmp = np.stack(noise_EEG_tmp,axis=0)
    
    print("clean",np.shape(clean))
    print("noise_EEG_tmp",np.shape(noise_EEG_tmp))
    
    EEG_tmp=[]
    for i in range(clean.shape[0]):
        v_tmp=[]
        for j in range(clean.shape[1]):
            h_tmp=[]
            h_tmp.extend(clean[i][j][0:1000])
            h_tmp.extend(noise_EEG_tmp[i][j][1000:2000])
            h_tmp.extend(clean[i][j][2000:3000])
            v_tmp.append(h_tmp)
        # print("h_tmp",np.shape(h_tmp))
        # print("v_tmp.shape",np.shape(v_tmp))
        EEG_tmp.append(v_tmp)
    
    print("EEG_tmp",np.shape(EEG_tmp))
    noise_EEG=np.array(EEG_tmp)
    # clean_ds=np.dsplit(clean,1000)
    
    # print("clean_ds.shape:",np.shape(clean_ds))
    # noise_EEG_tmp=np.dsplit(noise_EEG_tmp,1000)
    # # noise_EEG_tmp=noise_EEG_tmp.tolist()
    # print("noise_EEG_tmp.shape:",np.shape(noise_EEG_tmp))

    # noise_EEG.extend(clean_ds[:][:][:][0])
    # noise_EEG.extend(noise_EEG_tmp[:][:][:][1])
    # noise_EEG.extend(clean_ds[:][:][:][2])
      
    # print(np.shape(noise_EEG)) 
    return noise_EEG

'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        Fold_label: [k-fold] list, each element is [N,C]
        Fold_len:   [k-fold] list
'''


def get_isruc(data_folder, noise_type = 'EOG'):
    fold_label = []
    fold_clean = []
    fold_contaminated = []
    fold_len = []
    
    path_RawData = path.join(data_folder,'ISRUC_S3','RawData')
    path_Extracted = path.join(data_folder,'ISRUC_S3','ExtractedChannels')

    # all_channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1', 'LOC_A2', 'ROC_A1','X1', 'X2']
    clean_channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']
    if noise_type == 'EOG':
        noise_channels = [ 'LOC_A2', 'ROC_A1' ]
            
    for sub in range(1, 11):

        label = read_label(path_RawData, sub)

        psg_clean = read_psg(path_Extracted, sub, clean_channels)

        noise = read_noise(path_Extracted, sub, noise_channels)

        contaminated_signals = contaminate(psg_clean, noise)

        # print('Subject', sub, ':', label.shape, psg_clean.shape, noise.shape)
        
        assert len(label) == len(psg_clean)

        # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM

        label[label==5] = 4  # make 4 correspond to REM
        fold_label.append(np.eye(5)[label])

        fold_clean.append(psg_clean)
        fold_contaminated.append(contaminated_signals)
        fold_len.append(len(label))

        print('Read subject', sub, psg_clean.shape)
    
    return  fold_clean, fold_contaminated, fold_len