import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from time import time

from get_isruc import *
from data_prepare import *
from args import get_args
from model import *
from json import dumps
# from tensorboardX import SummaryWriter
import copy

#t-rrmse
def denoise_loss_mse(denoise, clean):      
    loss = (denoise-clean)**2
    return torch.mean(loss)

def denoise_loss_rmse(denoise, clean):      #tmse
    loss = (denoise-clean)**2
    return torch.sqrt(torch.mean(loss))

def denoise_loss_rrmset(denoise, clean):      #tmse

    rmse1 = denoise_loss_rmse(denoise, clean)
    rmse2 = denoise_loss_rmse(clean, torch.zeros_like(clean).to(clean.device))
    #loss2 = tf.losses.mean_squared_error(noise, clean)
    return rmse1/rmse2

def get_corr(pred, label):  #person cc
   
    pred_mean, label_mean = torch.mean(pred,dim=-1,keepdim=True), torch.mean(label,dim=-1,keepdim=True)
    
    corr = ( torch.mean((pred - pred_mean) * (label - label_mean), dim=-1, keepdim=True) ) / (
                torch.sqrt(torch.mean((pred - pred_mean) ** 2, dim=-1, keepdim=True)) * torch.sqrt(torch.mean((label - label_mean) ** 2, dim=-1, keepdim=True )))
    
    return torch.mean(corr)


def main(args):
    device = torch.device(  'cuda:{}'.format(args.cuda)  )
    # Set random seed
    utils.seed_torch(seed = args.rand_seed)
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    # Save superpara: args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    # tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    # Build dataset
    log.info('Building dataset...')

    if args.dataset == 'ISRUC':
        fold_clean, fold_contaminated, fold_len = get_isruc(args.data_dir)

    in_channels = fold_clean[0].shape[1]
    if args.do_train:
        
        DataGenerator = kFoldGenerator(fold_clean, fold_contaminated, args.batch_size)
        
        histories = []

        for i in range(10):
            # Train
            train_loader, val_loader = DataGenerator.getFold(i)
            print(128*'-')
            log.info('fold {} is running'.format(i+1))

            model = make_model( args=args, in_channels=in_channels ,DEVICE=device )
            total_param = 0
            for param_tensor in model.state_dict():
                total_param += np.prod(model.state_dict()[param_tensor].size())
            print('Net\'s total params:', total_param, flush=True)

            history = train(model, train_loader,val_loader, args, device, args.save_dir, log)
            
            histories.append( history )

            del model,train_loader,val_loader

            if i==0:
                fit_rrmse = np.array(history['rrmset_clean'])*fold_len[i]
                fit_corr = np.array(history['corr'])*fold_len[i]
                
            else:
                fit_rrmse = fit_rrmse + np.array(history['rrmset_clean'])*fold_len[i]
                fit_corr = fit_corr + np.array(history['corr'])*fold_len[i]
                           
    elif ( not args.do_train ) and args.load_model_path is None:
        raise ValueError( 'For fine-tuning, provide pretrained model in load_model_path!' )
    print('rrmse , corr')
    for idx in range(len(histories)):
        string_line = 'fold {}:'.format( idx )
        history = histories[idx]
        for k in history.keys():
            # f" | {mae:<7.2f}{rmse:<7.2f}{mape:<7.2f}{picp:<7.2f}{intervals:<7.2f}{mis:<7.2f}{test_cost_time:<6.2f}s"
            string_line += f'\t {history[k]:<.3f}'
        print( string_line )

    fit_rrmse     = fit_rrmse/np.sum(fold_len)
    fit_corr     = fit_corr/np.sum(fold_len)
    print(f'total rrmse:{fit_rrmse:<7.3f}. corr:{fit_corr:<7.3f}')
    print(128 * '_')
    print('End of training DenoiseNet')
    print(128 * '#')

def train(model, train_loader,val_loader, args, device, save_dir, log): #, tbx
    """
    Perform training and evaluate on val set
    """
    # Get saver
    saver = utils.CheckpointSaver(save_dir,log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.RMSprop(params=model.parameters(),
                           lr=args.lr_init)

    # Train
    epoch = 0
    message =  f"{'-' * 5} | {'-' * 7}{'Training'}{'-' * 7} | {'-' * 7}{'Validation'}{'-' * 7} |"
    print(  message )
    while (epoch != args.num_epochs):
        epoch += 1
        train_start_time = time()
        total_loss = 0

        total_samples = len(train_loader)

        for noiseeeg_batch, cleaneeg_batch in train_loader:
            noiseeeg_batch = noiseeeg_batch.to(device)
            cleaneeg_batch = cleaneeg_batch.to(device)
            
            denoiseoutput = model(noiseeeg_batch)
           
            loss = denoise_loss_mse(denoiseoutput, cleaneeg_batch)
            
            total_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss = total_loss/total_samples

        message = f"{epoch:<5} | {total_loss:<7.3f}{time() - train_start_time:<7.2f}s "

        print('\r' + message, end='', flush=False)

        if epoch % args.eval_every == 0:
            rrmset_clean, corr = evaluate(model,
                                        val_loader,
                                        args,
                                        device,
                                        save_dir,
                                        is_test=True)

    return {
        'rrmset_clean':rrmset_clean, 
        'corr':corr, 
    }

def evaluate(
        model,
        dataloader,
        args,
        DEVICE,
        save_dir,
        is_test=False,
        ):
    # To evaluate mode
    model.eval()

    val_losses = []
    y_pred_all = []
    y_true_all = []
    with torch.no_grad():
        val_start_time = time()
        for noiseeeg_batch, cleaneeg_batch in dataloader:
            noiseeeg_batch = noiseeeg_batch.to(DEVICE)
            cleaneeg_batch = cleaneeg_batch.to(DEVICE)
            denoiseoutput = model(noiseeeg_batch)
            # Update loss
            loss = denoise_loss_mse(denoiseoutput, cleaneeg_batch)
            val_losses.append(loss.item())

            y_pred_all.append(denoiseoutput)
            y_true_all.append(cleaneeg_batch)
      
    y_pred_all = torch.cat( y_pred_all,dim=0)
    y_true_all = torch.cat( y_true_all,dim=0)
    
    rrmset_clean = denoise_loss_rrmset(y_pred_all, y_true_all)
    corr = get_corr(y_pred_all, y_true_all)
    message = f"  | {rrmset_clean:<7.3f}{corr:<7.3f}{time() - val_start_time:<6.2f}s"

    print(message, end='', flush=False)
    print()
    return rrmset_clean.detach().cpu().numpy(), corr.detach().cpu().numpy()

if __name__ == '__main__':
    main(get_args())
