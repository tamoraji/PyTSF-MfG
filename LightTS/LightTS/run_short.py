import os
import torch
from datetime import datetime
from experiments.exp_short import Exp_short
import argparse
import pandas as pd
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='LightTS on financial datasets')
### -------  dataset settings --------------
parser.add_argument('--dataset_name', type=str, default='exchange_rate', choices=['electricity', 'solar_AL', 'exchange_rate', 'traffic'])
parser.add_argument('--data', type=str, default='../datasets/short/exchange_rate.txt',
                    help='location of the data file')
parser.add_argument('--normalize', type=int, default=2)

### -------  device settings --------------
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')

### -------  input/output length settings --------------                                                                            
parser.add_argument('--window_size', type=int, default=168, help='input length')
parser.add_argument('--horizon', type=int, default=3, help='prediction length')
parser.add_argument('--single_step', type=int, default=0, help='only supervise the final step')
parser.add_argument('--lastWeight', type=float, default=1.0,help='Loss weight lambda on the final step')

### -------  training settings --------------  
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--num_nodes',type=int,default=8,help='number of nodes/variables')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=5e-3,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--save_path', type=str, default='exp/financial_checkpoints/')
parser.add_argument('--model_name', type=str, default='LightTS')
parser.add_argument('--single_seed', type=bool, default=False)

### -------  model settings --------------  
parser.add_argument('--hiddim', default=512, type=int, help='hidden dimension')# H, EXPANSION RATE
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--chunk_size', type=int, default=42)
parser.add_argument('--c_dim', type=int, default=40)

parser.add_argument('--output_path', type=str, default='outputs') # when using yaml, must be ''

args = parser.parse_args()



seeds = [3142, 2431, 4321, 1234, 1324]

if __name__ == '__main__':
    DATADIR = '..'

    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    Exp=Exp_short

    if args.train:
        _rse = []
        _rae = []
        _corr = []
        for seed in seeds:
            torch.manual_seed(seed)  # reproducible
            torch.cuda.manual_seed_all(seed)
            exp = Exp(args)
            data = exp._get_data(DATADIR)
            before_train = datetime.now().timestamp()
            print("===================Normal Start of {}{} with seed{}=========================".format(args.dataset_name, args.horizon, seed))
            normalize_statistic = exp.train(DATADIR)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
            print("===================Normal End of {}{} with seed{}=========================".format(args.dataset_name, args.horizon, seed))
            rse, rae, corr = exp.validate(data,data.test[0],data.test[1], evaluate=True)
            _rse.append(rse)
            _rae.append(rae)
            _corr.append(corr)
            if args.single_seed:
                break

        
        print("Final average result of {}_{}_lb{}_la{}: rse {} | corr {}".format(args.model_name, args.dataset_name, args.window_size, args.horizon, np.mean(_rse), np.mean(_corr)))
        print("Final best    result of {}_{}_lb{}_la{}: rse {} | corr {}".format(args.model_name, args.dataset_name, args.window_size, args.horizon, min(_rse),     max(_corr)))







