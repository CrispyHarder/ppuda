# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Decoders to predict 1D-4D parameter tensors given node embeddings.

"""


import numpy as np
import torch
import torch.nn as nn
from .mlp import MLP
from .layers import get_activation
from _dwp.classes import Decoder3x3

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ConvDecoder(nn.Module):
    def __init__(self,
                 in_features=64,
                 hid=(128, 256),
                 out_shape=None,
                 num_classes=None,
                 gen_noise = False,
                 var_init = 1,
                 mu_scale  = 1,
                 var_scale = 0.1,
                 train_noise=False):
        super(ConvDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.gen_noise = gen_noise
        #IMPORTANT NOTE:
        # in case of train_noise = False:  
        #   Noise is N(0,var_init^2) distributed 
        # in case of train_noise = True:
        #   Mu is initiated with N(0,mu_scale^2) and the std is initiated as N(var_init,var_scale^2)
        if gen_noise:
            mean_data = torch.zeros(8)
            var_data = torch.ones(8)*torch.sqrt(torch.ones(1)*var_init) 
            if train_noise:
                # if the noise parameters shall be learned, we want to initialise such that 
                # the init values differ ! 
                mean_data = torch.randn(8)*mu_scale
                var_data = torch.tensor([max(torch.sqrt(torch.ones(1)*1e-3),torch.sqrt(torch.randn(1)*var_scale+var_init)) for i in range(8)])#torch.randn(8)*torch.sqrt(torch.ones(1)*0.1)+torch.ones(8)
            in_features += 8
            self.gen_means = torch.nn.Parameter(data=mean_data, requires_grad=train_noise)
            self.gen_vars = torch.nn.Parameter(data=var_data, requires_grad=train_noise)
        self.fc = nn.Sequential(nn.Linear(in_features,
                                          hid[0] * np.prod(out_shape[2:])),
                                nn.ReLU())
        conv = []
        for j, n_hid in enumerate(hid):
            n_out = np.prod(out_shape[:2]) if j == len(hid) - 1 else hid[j + 1]
            conv.extend([nn.Conv2d(n_hid, n_out, 1),
                         get_activation(None if j == len(hid) - 1 else 'relu')])
        self.conv = nn.Sequential(*conv)
        self.class_layer_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_shape[0], num_classes, 1))


    def forward(self, x, max_shape=(0,0), class_pred=False):

        N = x.shape[0]
        if self.gen_noise:
            noise = torch.randn(N,8).to(x.device)
            noise = torch.add(torch.mul(noise,self.gen_vars**2),self.gen_means)
            x = torch.cat([x,noise],dim=1)
        x = self.fc(x).view(N, -1, *self.out_shape[2:]) # N,128,11,11
        out_shape = self.out_shape
        if sum(max_shape) > 0:
            x = x[:, :, :max_shape[0], :max_shape[1]]
            out_shape = (out_shape[0], out_shape[1], max_shape[0], max_shape[1])
        x = self.conv(x).view(N, *out_shape)  # N, out, in, h, w
        if class_pred:
            x = self.class_layer_predictor(x[:, :, :, :, 0])  # N, num_classes, 64, w
            x = x[:, :, :, 0]  # N, num_classes, 64

        return x

class Conv3Decoder(nn.Module):
    def __init__(self,
                 in_features=32,
                 latent_dim = 2,
                 hidden_dim = 16,
                 out_shape=None,
                 num_classes=None):
        super(Conv3Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(in_features,self.latent_dim*np.prod(out_shape[:2])), # two-dim output for 64^2 slices
                                nn.ELU())
        self.conv = Decoder3x3(self.latent_dim,self.hidden_dim)
        self.class_layer_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_shape[0], num_classes, 1))


    def forward(self, x, max_shape=None, class_pred=False):

        N = x.shape[0]
        x = self.fc(x).view(N * np.prod(self.out_shape[:2]), self.latent_dim, 1, 1) # N*64^2,lat_dim, 1, 1
        out_shape = self.out_shape
        x = self.conv(x).view(N,*out_shape)  # N*64^2,1,3,3 -> view (N,64,64,3,3)
        if class_pred:
            x = self.class_layer_predictor(x[:, :, :, :, 0])  # N, num_classes, 64, 3
            x = x[:, :, :, 0]  # N, num_classes, 64
        return x

class MLPDecoder(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(64,),
                 out_shape=None,
                 num_classes=None):
        super(MLPDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.mlp = MLP(in_features=in_features,
                       hid=(*hid, np.prod(out_shape)),
                       activation='relu',
                       last_activation=None)
        self.class_layer_predictor = nn.Sequential(
            get_activation('relu'),
            nn.Linear(hid[0], num_classes * out_shape[0]))


    def forward(self, x, max_shape=(0,0), class_pred=False):
        if class_pred:
            x = list(self.mlp.fc.children())[0](x)  # shared first layer
            x = self.class_layer_predictor(x)  # N, 1000, 64, 1
            x = x.view(x.shape[0], self.num_classes, self.out_shape[1])
        else:
            x = self.mlp(x).view(-1, *self.out_shape)
            if sum(max_shape) > 0:
                x = x[:, :, :, :max_shape[0], :max_shape[1]]
        return x
