import os
import numpy as np
import time
import argparse


data_dir = '/data/Liulei/preprocess/data'
save_dir = 'result'

device = 2
num_epochs = 50

com_code = f'python main.py --cuda {device} --data_dir {data_dir}  --save_dir {save_dir}  --do_train --num_epochs {num_epochs} '

start_time = time.asctime(time.localtime(time.time()))
os.system(com_code)
print('\nstart  at',start_time)
print('finish at',time.asctime(time.localtime(time.time())))