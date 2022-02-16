import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
import PIL
import pandas as pd 
from PIL import Image
import torchvision.transforms as transforms 
from torch.utils.data import random_split
from torch import distributions as dist
import torch.nn.functional as F
import torchvision
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import yaml
import pickle
DATA_ROOT = os.path.join('data')# os.environ['DATA_ROOT']


def load_ghn_noise(path, device=None):

    from ppuda.ghn.nn import GHN2
    from ppuda.ghn.decoder import ConvDecoder

    with open(os.path.join(path, 'params.yaml')) as f:
        args = yaml.full_load(f)

    ghn = GHN2('cifar10')
    hid = ghn.hid
    var_init = args['var_init']
    mu_scale = args['mu_scale']
    var_scale = args['var_scale']
    train_noise = args['train_noise']
    
    ghn.decoder = ConvDecoder(in_features=hid,
                        hid=(hid * 4, hid * 8),
                        out_shape=(64,64,3,3),
                        num_classes=10,
                        gen_noise = True,
                        var_init = var_init,
                        mu_scale  = mu_scale,
                        var_scale = var_scale,
                        train_noise= train_noise)

    if device:
        ghn.load_state_dict(torch.load(os.path.join(path, 'ghn_params.torch'),map_location=device))
    else:
        ghn.load_state_dict(torch.load(os.path.join(path, 'ghn_params.torch')))
    return ghn.to(device)

def load_ghn_base(path, device=None):

    from ppuda.ghn.nn import GHN2
    from ppuda.ghn.decoder import ConvDecoder

    with open(os.path.join(path, 'params.yaml')) as f:
        args = yaml.full_load(f)

    ghn = GHN2('cifar10')
    hid = ghn.hid
    
    ghn.decoder = ConvDecoder(in_features=hid,
                        hid=(hid * 4, hid * 8),
                        out_shape=(64,64,3,3),
                        num_classes=10,
                        gen_noise = False)

    if device:
        ghn.load_state_dict(torch.load(os.path.join(path, 'ghn_params.torch'),map_location=device))
    else:
        ghn.load_state_dict(torch.load(os.path.join(path, 'ghn_params.torch')))
    return ghn.to(device)

def load_ghn_default(path, device=None):

    from ppuda.ghn.nn import GHN2

    ghn = GHN2('cifar10').to(device)
    
    return ghn

def bn_warm_start(data,net,device):
    net.train()
    for i,(x,_) in enumerate(data):
        x = x.to(device)
        net(x)
    return

class ConvDataset(Dataset):
    def __init__(self, file=None, data=None):
        super(ConvDataset, self).__init__()
        if file is not None:
            self.data = np.load(file)
        elif data is not None:
            self.data = np.copy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


def get_dataloaders(file, train_bs, test_bs, train_size=0.8):
    train, test = train_test_split(np.load(file), train_size=train_size)
    D = train.shape[-1]
    train, test = ConvDataset(data=train), ConvDataset(data=test)
    trainloader = torch.utils.data.DataLoader(train, batch_size=train_bs)
    testloader = torch.utils.data.DataLoader(test, batch_size=test_bs)

    return trainloader, testloader, D

class Pcam_dataset(Dataset):
    def __init__(self, data_dir, transform,data_type="train"):      
    
        # Get Image File Names
        cdm_data=os.path.join(data_dir,data_type)  # directory of files
        file_names = os.listdir(cdm_data) # get list of images in that directory
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names]   # get the full path to images
        
        # Get Labels
        labels_data=os.path.join(data_dir,"train_labels.csv") # labels are in a csv file named train_labels.csv
        labels_df=pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True) # set data frame index to id
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names] # obtain labels from df
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames) # size of dataset
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image) # Apply Specific Transformation to Image
        return image, self.labels[idx]

def load_cifar10_loaders(bs, test_bs):
    data_root = DATA_ROOT
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    #split train in train and val set
    n = len(trainset)
    len_val = int(0.1*n)
    len_train = n-len_val
    trainset, valset =random_split(trainset,[len_train,len_val], generator=torch.Generator().manual_seed(42))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=test_bs, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=0)

    return trainloader, valloader, testloader

class Pcam_dataset(Dataset):
    def __init__(self, data_dir, transform,data_type="train"):      
    
        # Get Image File Names
        cdm_data=os.path.join(data_dir,data_type)  # directory of files
        file_names = os.listdir(cdm_data) # get list of images in that directory
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names]   # get the full path to images
        self.images = []
        for name in self.full_filenames:
            image = Image.open(name)
            image = transform(image)
            self.images.append(image)
        self.images = torch.stack(self.images)
        # Get Labels
        labels_data=os.path.join(data_dir,"train_labels.csv") # labels are in a csv file named train_labels.csv
        labels_df=pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True) # set data frame index to id
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names] # obtain labels from df
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames) # size of dataset
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = self.images[idx]
        return image, self.labels[idx]

def load_pcam_loaders(train_bs,test_bs):
    # get general data, is not splitted or transformed yet
    data_dir = os.path.join(DATA_ROOT,'pcam')
    data_transformer = transforms.Compose([transforms.ToTensor()])
    img_dataset = Pcam_dataset(data_dir,data_transformer,'train')
    #now split the data; we have 220025 images 
    len_train = 153600
    len_val = 25000
    len_test = 41425
    train_ts, val_ts ,test_ts=random_split(img_dataset,[len_train,len_val, len_test], generator=torch.Generator().manual_seed(42)) 
    # set the transformations 
    transf = transforms.Compose([transforms.ToTensor()])
    train_ts.transform=transf
    val_ts.transform=transf
    test_ts.transform = transf

    trainloader = torch.utils.data.DataLoader(train_ts, batch_size=train_bs, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_ts, batch_size=test_bs, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_ts, batch_size=test_bs, shuffle=False, num_workers=0)
    return trainloader,valloader,testloader
    
def load_dataset(data, train_bs, test_bs, num_examples=None, augmentation=True, data_root=DATA_ROOT,
                 shuffle=True, seed=42):
    transform_train = transforms.Compose([
        MyPad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if data == 'cifar':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                                transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            a, _, b, _ = train_test_split(trainset.train_data, trainset.train_labels,
                                          train_size=num_examples, random_state=42)
            trainset.train_data = a
            trainset.train_labels = b
    elif data == 'pcam':
        # get general data, is not splitted or transformed yet
        data_dir = os.path.join(DATA_ROOT,'pcam')
        data_transformer = transforms.Compose([transforms.ToTensor()])
        img_dataset = Pcam_dataset(data_dir,data_transformer,'train')
        #now split the data
        len_img=len(img_dataset)
        len_train=int(0.8*len_img)
        len_val=len_img-len_train
        train_ts,val_ts=random_split(img_dataset,[len_train,len_val], generator=torch.Generator().manual_seed(seed)) # random split 80/20
        # set the transformations 
        tr_transf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.5),  
            transforms.RandomRotation(45),         
            transforms.RandomResizedCrop(96,scale=(0.8,1.0),ratio=(1.0,1.0)),
            transforms.ToTensor()])
        val_transf = transforms.Compose([transforms.ToTensor()])
        train_ts.transform=tr_transf
        val_ts.transform=val_transf
        trainset = train_ts
        testset = val_ts
    elif data == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True,
                                                 transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            a, _, b, _ = train_test_split(trainset.train_data, trainset.train_labels,
                                          train_size=num_examples, random_state=42)
            trainset.train_data = a
            trainset.train_labels = b
    elif data == 'svhn':
        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True,
                                             transform=transform_train if augmentation else transform_test)
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            a, _, b, _ = train_test_split(trainset.data, trainset.labels,
                                          train_size=num_examples, random_state=42)
            trainset.data = a
            trainset.labels = b
    elif data == 'mnist':
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)

        if num_examples is not None and num_examples != len(trainset):
            idxs, _ = train_test_split(np.arange(len(trainset)), train_size=num_examples, random_state=seed,
                                       stratify=tonp(trainset.train_labels))
            trainset.train_data = trainset.train_data[idxs]
            trainset.train_labels = trainset.train_labels[idxs]
    elif data == 'cifar5':
        CIFAR5_CLASSES = [0, 1, 2, 3, 4]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES,
                         random_labeling=False)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES,
                        random_labeling=False)
    elif data == 'not-mnist':
        trainset = torchvision.datasets.MNIST(root=os.path.join(data_root, 'not-mnist'), train=True,
                                              download=True, transform=transform_test)
        testset = torchvision.datasets.MNIST(root=os.path.join(data_root, 'not-mnist'), train=False,
                                             download=True, transform=transform_test)
        if num_examples is not None and num_examples != len(trainset):
            idxs, _ = train_test_split(np.arange(len(trainset)), train_size=num_examples, random_state=seed,
                                       stratify=tonp(trainset.train_labels))
            trainset.train_data = trainset.train_data[idxs]
            trainset.train_labels = trainset.train_labels[idxs]
    elif data == 'cifar5-rest':
        CIFAR5_CLASSES = [5, 6, 7, 8, 9]
        trainset = CIFAR(root=data_root, train=True, download=True,
                         transform=transform_train if augmentation else transform_test, classes=CIFAR5_CLASSES)
        testset = CIFAR(root=data_root, train=False, download=True, transform=transform_test, classes=CIFAR5_CLASSES)
    elif data == 'shapes':
        train_images = np.load(os.path.join(data_root, 'four-shapes/dataset/train_images.npy'))
        test_images = np.load(os.path.join(data_root, 'four-shapes/dataset/test_images.npy'))
        train_labels = np.load(os.path.join(data_root, 'four-shapes/dataset/train_labels.npy'))
        test_labels = np.load(os.path.join(data_root, 'four-shapes/dataset/test_labels.npy'))

        if num_examples != 4000:
            RuntimeWarning('==> --num-examples for shapes dataset should be 4000 <==')

        if num_examples is not None:
            train_images, _, train_labels, _ = train_test_split(train_images, train_labels, train_size=num_examples,
                                                                random_state=seed, stratify=train_labels)

        train_images, test_images = map(torch.Tensor, [train_images, test_images])
        train_labels, test_labels = map(torch.LongTensor, [train_labels, test_labels])

        trainset = torch.utils.data.TensorDataset(train_images, train_labels)
        testset = torch.utils.data.TensorDataset(test_images, test_labels)
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=shuffle, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=0)

    return trainloader, testloader


def pad(img, size, mode):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.pad(img, [(size, size), (size, size), (0, 0)], mode)


class MyPad(object):
    def __init__(self, size, mode='reflect'):
        self.mode = mode
        self.size = size
        self.topil = transforms.ToPILImage()

    def __call__(self, img):
        return self.topil(pad(img, self.size, self.mode))


def get_dataloader(file, bs, shuffle=False):
    data = np.load(file)
    D = data.shape[-1]
    data = ConvDataset(data=data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=shuffle)
    return dataloader, D


class FFGKL(nn.Module):
    """KL divergence between standart normal prior and fully-factorize gaussian posterior"""

    def __init__(self):
        super(FFGKL, self).__init__()

    def forward(self, mu, var):
        return -0.5 * (1 + torch.log(var) - mu.pow(2) - var).sum()


class VAEELBOLoss(nn.Module):
    """docstring for ELBOLoss"""

    def __init__(self, likelihood=None, kl=None, use_cuda=False):
        super(VAEELBOLoss, self).__init__()
        if likelihood is None:
            self.likelihood = NormalLikelihood()
        if kl is None:
            self.kl = FFGKL()
        if use_cuda:
            self.likelihood = self.likelihood.cuda()
            self.kl = self.kl.cuda()

    def forward(self, target, likelihood_params, var_params):
        return self.likelihood(target, *likelihood_params), self.kl(*var_params)


class NormalLikelihood(nn.Module):
    def __init__(self):
        super(NormalLikelihood, self).__init__()

    def forward(self, target, mu, var):
        loss = torch.sum(-(target - mu)**2 / var - np.log(2 * np.pi) - torch.log(var)) * 0.5
        return loss


def tonp(x):
    return x.cpu().detach().numpy()


def get_kernels(net):
    convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    weights = [tonp(m.weight) for m in convs]
    weights = [w.reshape((-1, w.shape[-2], w.shape[-1])) for w in weights]
    return weights

def weight_init(module, initf):
    def foo(m):
        classname = m.__class__.__name__.lower()
        if isinstance(m, module):
            initf(m.weight)
    return foo


def const_init(val):
    def foo(var):
        nn.init.constant_(var, val)
    return foo


def vae_init(vae):
    def foo(var):
        shape = var.shape
        z = torch.randn(shape[0] * shape[1], vae.encoder.z_dim, 1, 1).to(vae.device)
        w, _ = vae.decode(z)
        var.data = w.detach().view(*shape)
    return foo


def flow_init(flow):
    def foo(var):
        shape = var.shape
        k = var.shape[-1]
        w = flow.sample(shape[0] * shape[1]).view(shape[0] * shape[1], 1, k, k)
        var.data = w.detach().view(*shape)
    return foo


def pretrained_init(net):
    weights = torch.cat([m.weight.view((-1, 3, 3)).detach() for m in net.modules() if isinstance(m, nn.Conv2d)])
    N = weights.shape[0]

    def foo(var):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(var)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        M = np.prod(var.shape[:2])
        idxs = np.random.choice(N, size=M, replace=True)
        w_std = torch.std(weights[idxs], unbiased=True).to(var.device)
        var.data = weights[idxs].view(*var.shape).to(var.device) * std / w_std

    return foo


class MovingMetric(object):
    def __init__(self):
        self.n = 0.
        self.val = 0.

    def add(self, v, n):
        self.n += n
        self.val += v

    def get_val(self):
        return self.val / max(self.n, 1)

class AvgrageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = np.sum(self.sum) / self.cnt


def mc_ensemble(net, dataloader, n_tries=10, log=False):
    gt = []
    pred = []
    for x, y in dataloader:
        x = x.to(net.device)
        gt.append(y.numpy())
        ens = None
        for i in range(n_tries):
            p = tonp(F.log_softmax(net(x), dim=1))
            if ens is None:
                ens = p
            else:
                w = np.array([1 if i > 1 else (1. / n_tries), 1./n_tries])[:, np.newaxis, np.newaxis]
                ens = logsumexp(np.stack([ens, p]), axis=0, b=w)
        pred.append(ens)

    if log:
        return np.concatenate(pred), np.concatenate(gt)

    return np.exp(np.concatenate(pred)), np.concatenate(gt)


def get_logp(net, dataloader):
    gt = []
    logits = []
    for x, y in dataloader:
        x = x.to(net.device)
        gt.append(y.numpy())
        p = tonp(F.log_softmax(net(x)))
        logits.append(p)

    return np.concatenate(logits), np.concatenate(gt)


class CIFAR(torchvision.datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset with several classes.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, classes=None, random_labeling=False):

        if classes is None:
            classes = np.arange(10).tolist()

        self.classes = classes[:]

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.random_labeling = random_labeling

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            mask = np.isin(self.train_labels, classes)
            self.train_labels = [classes.index(l) for l, cond in zip(self.train_labels, mask) if cond]
            if self.random_labeling:
                self.train_labels = np.random.permutation(self.train_labels)

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))[mask]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            mask = np.isin(self.test_labels, classes)
            self.test_labels = [classes.index(l) for l, cond in zip(self.test_labels, mask) if cond]

            self.test_data = self.test_data.reshape((10000, 3, 32, 32))[mask]
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
