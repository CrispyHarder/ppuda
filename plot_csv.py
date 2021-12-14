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
args = parser.parse_args()

logs_path = os.path.join('logs',f'exman-{args.train_script}.py','runs')
run = [os.path.join(logs_path,run) for run in os.listdir(logs_path) if run.startswith(args.run)][0]

METR_NAMES = ['time','train_loss','train_top1','val_top1','val_res_top1','val_loss','train_top5','val_top5']
SAVE_PATH = os.path.join('..','..','small-results',str(date.today()),args.train_script,run)
SAVE_SPEC = 'last 20'
SAVE_PLOTS = True
SHOW_PLOTS = True
STARTING_AT = 100
ENDING_AT = 121

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

file  = os.path.join(run,f'{args.train_script}.py-logs.csv')

data = np.genfromtxt(file,delimiter=',')[:,1:8]

for i,m_name in enumerate(METR_NAMES):
    plt.figure(i)
    plt.plot(data[STARTING_AT:ENDING_AT,i])
    plt.title('{} convergence'.format(run))
    if SAVE_PLOTS:    
        plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+m_name))
    if SHOW_PLOTS:
        plt.show()