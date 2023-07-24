import shutil
import torch 
import models.resnet as resnet
import yaml
import os 
import re
import warnings
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets import ImageFolder
from torchvision import transforms


class MultistepMultiGammaLR(_LRScheduler):
    """Decays the learning rate of each parameter group by a different gamma,
    (which can be different every time in this adaption of MultiStepLR) once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (list(float)): Multiplicative factor of learning rate decay.
            One for every milestone
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example: with same gamma every time 
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, milestones, gamma, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.counter = -1
        super(MultistepMultiGammaLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        
        if not self._get_lr_called_within_step: # pylint: disable=no-member
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        #if we change lr, increase counter by one
        self.counter += 1
        return [group['lr'] * self.gamma[self.counter]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        lr_modifier_total = 1
        for i in range(self.counter):
            lr_modifier_total = lr_modifier_total * self.gamma[i]
        return [base_lr * self.gamma ** lr_modifier_total
                for base_lr in self.base_lrs]
                

def get_state_dict_from_checkpoint(checkpoint_path, map_location=None):
    '''loads the state dict from a given checkpoint path'''
    if map_location:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    elif torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint['state_dict']

def load_resnet_from_checkpoint(checkpoint_path, model_type, dataset_name):
    '''Gets a path to a checkpoint and a model type and loads the model 
    using the state dict'''

    model = resnet.__dict__[model_type]()
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint_path))
    if torch.cuda.is_available():
        model.cuda()  
    return model

def save_args_params(args, dir_path):
    dumbd = args.__dict__.copy()
    path = os.path.join(dir_path,'params.yaml')
    if os.path.exists(path):
        os.remove(path)
    with open(path,'a') as f:
        yaml.dump(dumbd,f)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def get_pcam_transfer_dataloaders(bs,n_ds=""):
    base_dir = f"/gris/gris-f/homestud/charder/deep-weight-prior/data/pcam_folders{n_ds}"

    train_dir = os.path.join(base_dir,"train")
    valid_dir = os.path.join(base_dir,"valid")
    test_dir = os.path.join(base_dir,"test")

    train_ds = ImageFolder(root=train_dir,transform=transforms.ToTensor())
    valid_ds = ImageFolder(root=valid_dir,transform=transforms.ToTensor())
    test_ds = ImageFolder(root=test_dir,transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=6)
    valloader = torch.utils.data.DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=6)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=6)
    
    return trainloader,valloader,testloader