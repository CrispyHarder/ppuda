# File to filter out runs using e.g. specific initialisations and comparing the 
# convergence of multiple models 

import os 
import pandas
import numpy as np
import matplotlib.pyplot as plt 
import yaml 
from datetime import date
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-r','--run',type=str)
parser.add_argument('-t','--train_script',default='train_ghn_self')
parser.add_argument('-p','--no_print', action='store_false', default=True)
parser.add_argument('-s','--no_save', action='store_false', default=True)
args = parser.parse_args()

logs_path = os.path.join('logs',f'exman-{args.train_script}.py','runs')
run = [os.path.join(logs_path,run) for run in os.listdir(logs_path) if run.startswith(args.run)][0]

METR_NAMES = ['time','train_loss','train_top1','val_top1','val_res_top1','val_loss','train_top5','val_top5']
SAVE_PATH = os.path.join('..','..','small-results',str(date.today()),args.train_script,args.run)
SAVE_SPEC = ''
SAVE_PLOTS = args.no_save
SHOW_PLOTS = args.no_print


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

file  = os.path.join(run,f'{args.train_script}.py-logs.csv')

data = np.genfromtxt(file,delimiter=',')[:,1:9]
epochs = np.shape(data)[0]

for i,m_name in enumerate(METR_NAMES):
    plt.figure(i)
    plt.plot(np.arange(epochs),data[:,i])
    plt.title('{} {}'.format(args.run,m_name))
    if SAVE_PLOTS:    
        plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+m_name))
    if SHOW_PLOTS:
        plt.show()