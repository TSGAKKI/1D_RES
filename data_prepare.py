import sklearn.model_selection as ms
import numpy as np
import scipy.io as sio
import math
import torch
from tqdm import tqdm

def get_rms(records):   
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

def random_signal(signal,combin_num):
    # Random disturb and augment signal
    random_result=[]

    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0],signal.shape[1])
        random_result.append(shuffled_dataset)
    random_result  = np.array(random_result)
    return  random_result

class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, fold_clean, fold_contaminated, batch_size ):
        if len(fold_clean) != len(fold_contaminated):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(fold_contaminated)
        self.clean_list = fold_clean
        self.noise_list = fold_contaminated
        self.batch_size = batch_size

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        for p in range(self.k):
            if p != i:
                if isFirst:
                    train_clean = self.clean_list[p]
                    train_contaiminated = self.noise_list[p]
                    isFirst = False
                else:
                    train_clean = np.concatenate((train_clean, self.clean_list[p]))
                    train_contaiminated = np.concatenate((train_contaiminated, self.noise_list[p]))
                    
            else:
                val_clean = self.clean_list[p]
                val_contaminated = self.noise_list[p]
        
        train_clean = torch.from_numpy(train_clean).type(torch.FloatTensor)#.to(DEVICE)  
        train_contaiminated = torch.from_numpy(train_contaiminated).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
        train_dataset = torch.utils.data.TensorDataset(train_contaiminated, train_clean)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_clean = torch.from_numpy(val_clean).type(torch.FloatTensor)#.to(DEVICE)  
        val_contaminated = torch.from_numpy(val_contaminated).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
        val_dataset = torch.utils.data.TensorDataset(val_contaminated, val_clean)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

def prepare_data(DEVICE, batch_size,EEG_all, noise_all, combin_num, train_per, noise_type):
    # Here we use eeg and noise signal to generate scale transed training, validation, test signal
    # print(EEG_all.shape) # 4514 512
    # print(noise_all.shape) # 3400 512
    # disorganize 
    EEG_all_random = np.squeeze(random_signal(signal = EEG_all, combin_num = 1))
    noise_all_random = np.squeeze(random_signal(signal = noise_all, combin_num = 1))  
    # print( (EEG_all_random>0).any())
    # exit()
    if noise_type == 'EMG':  # Training set will Reuse some of the EEG signal to much the number of EMG
        reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
        EEG_reuse = EEG_all_random[0 : reuse_num, :]
        EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
        print('EEG segments after reuse: ',EEG_all_random.shape[0])

    elif noise_type == 'EOG':  # We will drop some of the EEG signal to match the number of EMG
        EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]
        print('EEG segments after drop: ',EEG_all_random.shape[0])

    # get the 
    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0]) # the number of segmentations used in training process
    validation_num = round((EEG_all_random.shape[0] - train_num) / 2) # the number of segmentations used in validation process
    #test_num = EEG_all_random.shape[0] - train_num - validation_num  # Rest are the number of segmentations used in test process

    train_eeg = EEG_all_random[0 : train_num, :]
    validation_eeg = EEG_all_random[train_num : train_num + validation_num, :]
    test_eeg = EEG_all_random[train_num + validation_num : EEG_all_random.shape[0], :]

    train_noise = noise_all_random[0 : train_num, :]
    validation_noise = noise_all_random[train_num : train_num + validation_num,:]
    test_noise = noise_all_random[train_num + validation_num : noise_all_random.shape[0], :]

    EEG_train = random_signal(signal = train_eeg, combin_num = combin_num).reshape(combin_num * train_eeg.shape[0], timepoint)
    NOISE_train = random_signal(signal = train_noise, combin_num = combin_num).reshape(combin_num * train_noise.shape[0], timepoint)

    print('After random disturb EEG_train:',EEG_train.shape) # 2720 *10
    print('After random disturb NOISE_train:',NOISE_train.shape)
    
    #################################  simulate noise signal of training set  ##############################

    #create random number between -10dB ~ 2dB
    SNR_train_dB = np.random.uniform(-7, 2, (EEG_train.shape[0])) # 27200
    
    SNR_train = 10 ** (0.1 * (SNR_train_dB))

    # combin eeg and noise for training set 
    noiseEEG_train=[]
    NOISE_train_adjust=[]
    for i in range (EEG_train.shape[0]):
        eeg = EEG_train[i].reshape(EEG_train.shape[1]) # 512
        noise = NOISE_train[i].reshape(NOISE_train.shape[1])
       
        coe = get_rms(eeg)/(get_rms(noise)*SNR_train[i])
        noise = noise*coe
        neeg = noise + eeg

        NOISE_train_adjust.append(noise)
        noiseEEG_train.append(neeg)

    noiseEEG_train=np.array(noiseEEG_train)
    NOISE_train_adjust=np.array(NOISE_train_adjust)    

    # variance for noisy EEG
    EEG_train_end_standard = []
    noiseEEG_train_end_standard = []

    for i in range(noiseEEG_train.shape[0]):
        # Each epochs divided by the standard deviation
        eeg_train_all_std = EEG_train[i] / np.std(noiseEEG_train[i])
        EEG_train_end_standard.append(eeg_train_all_std)

        noiseeeg_train_end_standard = noiseEEG_train[i] / np.std(noiseEEG_train[i])
        noiseEEG_train_end_standard.append(noiseeeg_train_end_standard)

    noiseEEG_train_end_standard = np.array(noiseEEG_train_end_standard) # 输入
    EEG_train_end_standard = np.array(EEG_train_end_standard) # label
    print('training data prepared', noiseEEG_train_end_standard.shape, EEG_train_end_standard.shape )
    
    #################################  simulate noise signal of validation  ##############################

    SNR_val_dB = np.linspace(-7.0, 2.0, num=(10))
    
    SNR_val = 10 ** (0.1 * (SNR_val_dB))

    eeg_val = np.array(validation_eeg)
    noise_val = np.array(validation_noise)
    
    # combin eeg and noise for test set 
    EEG_val = []
    noise_EEG_val = []
    for i in range(10):
        
        noise_eeg_val = []
        for j in range(eeg_val.shape[0]):
            eeg = eeg_val[j]
            noise = noise_val[j]
            
            coe = get_rms(eeg) / (get_rms(noise) * SNR_val[i])
            noise = noise * coe
            neeg = noise + eeg
            
            noise_eeg_val.append(neeg)
        
        EEG_val.extend(eeg_val)
        noise_EEG_val.extend(noise_eeg_val)


    noise_EEG_val = np.array(noise_EEG_val)
    EEG_val = np.array(EEG_val)


    # std for noisy EEG
    EEG_val_end_standard = []
    noiseEEG_val_end_standard = []
    # std_VALUE = []
    for i in range(noise_EEG_val.shape[0]):
        
        # store std value to restore EEG signal
        std_value = np.std(noise_EEG_val[i])
        #std_VALUE.append(std_value)

        # Each epochs of eeg and neeg was divide by the standard deviation
        eeg_val_all_std = EEG_val[i] / std_value
        EEG_val_end_standard.append(eeg_val_all_std)

        noiseeeg_val_end_standard = noise_EEG_val[i] / std_value
        noiseEEG_val_end_standard.append(noiseeeg_val_end_standard)

    #std_VALUE = np.array(std_VALUE)
    noiseEEG_val_end_standard = np.array(noiseEEG_val_end_standard)
    EEG_val_end_standard = np.array(EEG_val_end_standard)
    print('validation data prepared, validation data shape: ', noiseEEG_val_end_standard.shape, EEG_val_end_standard.shape)

    #################################  simulate noise signal of test  ##############################

    SNR_test_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_test = 10 ** (0.1 * (SNR_test_dB))

    eeg_test = np.array(test_eeg)
    noise_test = np.array(test_noise)
    
    # combin eeg and noise for test set 
    EEG_test = []
    noise_EEG_test = []
    for i in range(10):
        
        noise_eeg_test = []
        for j in range(eeg_test.shape[0]):
            eeg = eeg_test[j]
            noise = noise_test[j]
            
            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[i])
            noise = noise * coe
            neeg = noise + eeg
            
            noise_eeg_test.append(neeg)
        
        EEG_test.extend(eeg_test)
        noise_EEG_test.extend(noise_eeg_test)


    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)


    # std for noisy EEG
    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    std_VALUE = []
    for i in range(noise_EEG_test.shape[0]):
        
        # store std value to restore EEG signal
        std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(std_value)

        # Each epochs of eeg and neeg was divide by the standard deviation
        eeg_test_all_std = EEG_test[i] / std_value
        EEG_test_end_standard.append(eeg_test_all_std)

        noiseeeg_test_end_standard = noise_EEG_test[i] / std_value
        noiseEEG_test_end_standard.append(noiseeeg_test_end_standard)
    
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)
    print('test data prepared, test data shape: ', noiseEEG_test_end_standard.shape, EEG_test_end_standard.shape)

    train_x = torch.from_numpy(noiseEEG_train_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    train_target = torch.from_numpy(EEG_train_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_x = torch.from_numpy(noiseEEG_val_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    val_target = torch.from_numpy(EEG_val_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_target)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_x = torch.from_numpy(noiseEEG_test_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    test_target = torch.from_numpy(EEG_test_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_target)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader 

