'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import _dwp.utils as utils
import numpy as np 
from torch.autograd import Variable
import os 

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        block = BasicBlock

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def mult_weights_init(self, init_mode, init_root, device, dataset, prior=None):
        '''initialises the layers of the resnet
        Args: 
            init_mode (str): how to initialise´
            init_root (str): the path to the folder containing the layer directories
            device (torch.device): the device to load the vaes'''
        # Since the Resnets are always initialised with he- init at contruction,
        #  we dont have to do anything
        if init_mode == 'he':
            return

        if init_mode.startswith('ghn'):
            path = os.path.join('data','resnet20','global_init_models',dataset,init_mode)
            ghn = utils.load_ghn(path,device=device)
            self = ghn.forward(self,bn_train=False).to(device)
            return

        self.apply(utils.weight_init(module=nn.Conv2d, initf=nn.init.xavier_normal_))
        self.apply(utils.weight_init(module=nn.Linear, initf=nn.init.xavier_normal_))

        # get all conv layers 
        sd = self.state_dict()
        i = 0
        for params in sd:
            if 'conv' in params:
                init = init_mode
                w =  sd[params]
                short_path = os.path.join(init_root,'layer_{}'.format(i),init_mode)
                if init == 'vae':
                    vae_path = short_path
                    vae = utils.load_vae(vae_path,device=device)
                    z = torch.randn(w.size(0) * w.size(1), vae.encoder.z_dim, 1, 1).cuda()
                    x = vae.decode(z)[0]
                    sd[params] = x.reshape(w.shape)
                elif init == 'xavier':
                    pass
                elif init.startswith('vqvae1'):
                    vqvae_path = short_path
                    vqvae = utils.load_vqvae1(vqvae_path,device=device)
                    if prior.startswith('pixelcnn'):
                        pixel_path = os.path.join(short_path,prior)
                        pixelcnn = utils.load_pixelcnn(pixel_path, device=device)
                        prior_sample = pixelcnn.generate(batch_size=w.size(0) * w.size(1),device=device)
                        quantized = vqvae._vq_vae._embedding(prior_sample)
                        x = vqvae.decoder(quantized.permute(0, 3, 1, 2))
                    else:
                        x = vqvae.sample(w.size(0) * w.size(1),device=device)
                    sd[params] = x.reshape(w.shape)
                elif init.startswith('tvae'):
                    tvae_path = short_path
                    tvae = utils.load_tvae(tvae_path,device=device)
                    x = tvae.generate(batch_size=w.size(0) * w.size(1),device=device)
                    sd[params] = x.reshape(w.shape)
                elif init.startswith('lvae'):
                    lvae_path = short_path
                    lvae = utils.load_lvae(lvae_path,device=device)
                    x = lvae.generate(batch_size=w.size(0) * w.size(1),device=device)
                    sd[params] = x.reshape(w.shape)
                elif init == 'flow':
                    #deprecated 
                    flow_path = short_path
                    flow = utils.load_flow(flow_path, device=self.device)
                    utils.flow_init(flow)(w)   
                elif init == 'filters':
                    #deprecated
                    filters = np.load(os.path.join(short_path,'filters.torch'))
                    filters = np.concatenate([filters]*10)
                    N = np.prod(w.shape[:2])
                    filters = filters[np.random.permutation(len(filters))[:N]]
                    w.data = torch.from_numpy(filters.reshape(*w.shape)).to(self.device)
                elif init == 'recon':
                    #deprecated
                    filters = np.load(os.path.join(short_path,'filters.torch'))
                    filters = np.concatenate([filters]*10)
                    N = np.prod(w.shape[:2])
                    filters = filters[np.random.permutation(len(filters))[:N]]
                    vae_path = os.path.join(short_path,'vae_params.torch')
                    vae = utils.load_vae(vae_path, device=self.device)
                    filters = vae(torch.from_numpy(filters).to(self.device))[1][0]
                    w.data = filters.reshape_as(w)
                else:
                    raise NotImplementedError('no {} init'.format(init))
                i += 1
        self.load_state_dict(sd)    

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet([3, 3, 3])


def resnet32():
    return ResNet([5, 5, 5])


def resnet44():
    return ResNet([7, 7, 7])


def resnet56():
    return ResNet([9, 9, 9])


def resnet110():
    return ResNet([18, 18, 18])


def resnet1202():
    return ResNet([200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()