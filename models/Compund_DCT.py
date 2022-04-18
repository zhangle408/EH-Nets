"""
    CompundDCT_Conv Definition.

    Licensed under the BSD License [see LICENSE for details].

    Written by Yao Lu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as init
import torchvision
from models.SCFilter import SCFConv2d


def shuffle_channel(x, ni):
    x = x.view(x.size(0), ni, x.size(1)//ni, x.size(2),x.size(3))
    x = x.permute(0,2,1,3,4).contiguous()
    x = x.view(x.size(0), -1, x.size(3), x.size(4))
    return x
    

def dct_filters(k=3, compund_level=None, level=None, DC=True, groups=1,expand_dim=1):
    
    filter_bank = np.zeros((k,k, k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            for x in range(k):
                for y in range(k):
                    filter_bank[i, j, x, y] = math.cos((math.pi * (x + .5) * i) / k) * math.cos((math.pi * (y + .5) * j) / k)
            if (not compund_level is None): #l1_norm:
                filter_bank[i, j, :, :] /= np.sum(np.abs(filter_bank[i, j, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank[i, j, :, :] *= (2.0 / k) * ai * aj

    if (not compund_level is None):
        filter_bank_expand=np.zeros((2*k-1, 2*k-1,k,k), dtype=np.float32)
        for i in range(2*k-1):
            for j in range(2*k-1):
                if (i%2==0) and (j%2==0):
                    filter_bank_expand[i,j,:,:]=filter_bank[i//2,j//2,:,:]
                elif (i%2==0) and (j%2!=0):
                    filter_bank_expand[i,j,:,:k-1]=filter_bank[i//2,j//2,:,1:]
                    filter_bank_expand[i,j,:,k-1]=filter_bank[i//2,(j+1)//2,:,0]

                elif (i%2!=0) and (j%2==0):
                    filter_bank_expand[i,j,:k-1,:]=filter_bank[i//2,j//2,1:,:]
                    filter_bank_expand[i,j,k-1,:]=filter_bank[(i+1)//2,j//2,0,:]

                else:
                    filter_bank_expand[i,j,:k-1,:k-1]=filter_bank[i//2,j//2,1:,1:]#top-left
                    filter_bank_expand[i,j,k-1,k-1]=filter_bank[(i+1)//2,(j+1)//2,0,0]#bottom-right
                    filter_bank_expand[i,j,k-1,:k-1]=filter_bank[(i+1)//2,j//2,0,1:]#bottom-left
                    filter_bank_expand[i,j,:k-1,k-1]=filter_bank[i//2,(j+1)//2,1:,0]#Top-right

    
    if (not compund_level is None):
        k_compund=2*k-1
    else:
        k_compund=k
    if level is None:
        nf = k_compund**2 
    else:
        if level <= k_compund:
            nf = level*(level+1)//2 
        else:
            r = 2*k_compund-1 - level
            nf = k_compund**2 - r*(r+1)//2 
    filter_bank_compund = np.zeros((nf, k, k), dtype=np.float32)


    m=0
    for i in range(k_compund):
        for j in range(k_compund):
            if (not DC and i == 0 and j == 0) or (not level is None and i + j >= level):
                continue
            #if (not DC and i == 0 and j == 0) or (i+j)%2!=0:
                #continue
            for x in range(k):
                for y in range(k):
                    if (not compund_level is None):
                        filter_bank_compund[m, x, y] =filter_bank_expand[i, j, x, y]
                    else:
                        filter_bank_compund[m, x, y] =filter_bank[i, j, x, y]
            if (not compund_level is None):#l1_norm
                filter_bank_compund[m, :, :] /= np.sum(np.abs(filter_bank_compund[m, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank_compund[m, :, :] *= (2.0 / k) * ai * aj
            m += 1

    filter_bank_compund = np.tile(np.expand_dims(filter_bank_compund, axis=expand_dim), (groups, 1, 1, 1))
    return torch.FloatTensor(filter_bank_compund)

class CompundDCT_Conv(nn.Module):
    drop_filter_stage=0
    #weight_list=[]
    def __init__(self, ni, no, kernel_size, stride=1, padding=0, bias=True, dilation=1, use_bn=True, compund_level=1, level=3, DC=True, groups=1, last_rate=1., balance_weight=1e-4):
        super(CompundDCT_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.compund_dct = nn.Parameter(dct_filters(k=kernel_size, compund_level=compund_level, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level, DC=DC), requires_grad=False)
        self.ni = ni
        nf = self.compund_dct.shape[0] // ni if use_bn else self.compund_dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm2d(ni*nf)
            self.linear_DCT = SCFConv2d(in_channels=ni*nf, out_channels=no,bais=bias, kernel_size=1,  n_sc=(1./groups), fre_num=nf, last_rate=last_rate,  balance_weight= balance_weight)
            #self.linear_DCT = SCFConv2d(in_channels=ni*nf, out_channels=no,bais=bias, kernel_size=1,  n_lego=groups, fre_num=nf, last_rate=last_rate,  balance_weight= balance_weight)
            #self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups * nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        else:
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups, nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        self.bias =  None
        #print(self.compund_dct.shape)
    def forward(self, x):
        if not hasattr(self, 'bn'):
            filt = torch.sum(self.weight * self.compund_dct, dim=2)
            x = F.conv2d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            return x
        else:
            #print(self.groups)
            SCFConv2d.drop_filter_stage=CompundDCT_Conv.drop_filter_stage
            #SCFConv2d.weight_list=CompundDCT_Conv.weight_list
            x = F.conv2d(x, self.compund_dct, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=x.size(1))
            
            x = self.bn(x)
            #x = shuffle_channel(x, self.ni)
            x = x.view(x.size(0), self.ni, x.size(1)//self.ni, x.size(2), x.size(3)) #use conv3d 
            #print(x.size(),self.ni,self.compund_dct.shape)
            x = self.linear_DCT(x)
            return x

