import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import yaml 
from datetime import date
import seaborn as sns 
import json
import torch
import torch.nn as nn
import torch.nn.init as init
import _dwp.utils as utils
from torch.autograd import Variable
import random 
from models.resnet import ResNet
from _dwp.logger import Logger
import _dwp.myexman as myexman
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from _dwp.my_utils import get_pcam_transfer_dataloaders
from tqdm import tqdm 

run_numbers=[109,125]
device = torch.device("cuda")

def predict(data, net):
    pred = []
    l = []
    for x, y in tqdm(data):
        l.append(y.numpy())
        x = x.to(device)
        p = F.log_softmax(net(x), dim=1)
        pred.append(p.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(l)

def load_resnet(path,ds="pcam",device=torch.device("cuda")):
    n_classes = 10 if ds=='cifar' else 2
    model = ResNet([3,3,3],num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(path,map_location=device))
    model = model.to(device)
    return model

logs_path = "/gris/gris-f/homestud/charder/ppuda/logs/exman-/gris/gris-f/homestud/charder/ppuda/train_net_pcam_transfer.py/runs"
runs = [os.path.join(logs_path,run) for run in os.listdir(logs_path) if run_numbers[0]<=int(run[:6].lstrip("0"))<=run_numbers[1]]

trainloader, valloader, testloader = get_pcam_transfer_dataloaders(128)

i=1

for run in runs:

    # Print number of iteration
    print(i,"/",len(runs))
    i+=1

    # Load model
    sd_path = os.path.join(run,"net_params.torch")
    model = load_resnet(sd_path)

    # Make predictions
    model.eval()
    with torch.no_grad():
        pred, labels = predict(testloader,model)
        acc = np.mean(pred.argmax(1) == labels)
    
    # Write accuracy into a json file 
    acc_save_path = os.path.join(run,"test_acc.json")
    data = {"test_acc": acc}
    json_string = json.dumps(data, indent=4)  #     # Step 2: Convert the dictionary to a JSON string 'indent=4' adds indentation for readability
    with open(acc_save_path, "w") as file:
        file.write(json_string)
