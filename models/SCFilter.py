"""
    Shared Combinational Filters Definition.

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
from models.Discrete import DiscreteFunction

def shuffle_channel(x, ni):
    x = x.view(x.size(0), ni, x.size(1)//ni, x.size(2),x.size(3))
    x = x.permute(0,2,1,3,4).contiguous()
    x = x.view(x.size(0), -1, x.size(3), x.size(4))
    return x

def entropy_filter(x):
    abs_sum=x.abs().sum(dim=[1,2,3]).view(-1)
    return abs_sum

def select_filter(n_sc, drop_filter_stage, last_rate, fre_num, sc_filter):
    assert drop_filter_stage<=2
    new_last_rate=1.-(((1.-last_rate)/2)*drop_filter_stage)
    preserve_rate=[1.-(((1.-new_last_rate)/(fre_num-1))*i) for i in range(fre_num)]
    mask=torch.FloatTensor(np.ones((fre_num, n_sc,1,1,1), dtype=np.float32)).to(sc_filter.device)
    sc_filter_detach=sc_filter.detach()
    entropy_list=entropy_filter(sc_filter_detach)
    for i in range(fre_num):
        drop_filter=int((1.-preserve_rate[i])*n_sc)
        if drop_filter>0:
            topk_maxmum, _ = entropy_list.topk(int((1.-preserve_rate[i])*n_sc), dim=0, largest=False, sorted=False)
            clamp_max, _ = topk_maxmum.max(dim=0, keepdim=True)
            mask_index = entropy_list.le(clamp_max)
            mask[i][mask_index] = 0.
    return mask
class SCFConv2d(nn.Module):
    drop_filter_stage=0
    def __init__(self, in_channels, out_channels, bais, kernel_size,  n_sc, fre_num, last_rate, balance_weight):
        super(SCFConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.fre_num = in_channels, out_channels, kernel_size, fre_num
        self.basic_channels = in_channels // self.fre_num
        self.n_sc = int(self.out_channels * n_sc)
        self.fre_num = fre_num
        self.last_rate = last_rate
        self.balance_weight=balance_weight
        self.sc = nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.n_sc, self.basic_channels, self.kernel_size, self.kernel_size),mode='fan_out', nonlinearity='relu'))#m,N,3,3
        self.aux_coefficients = nn.Parameter(init.kaiming_normal_(torch.rand(self.fre_num, self.out_channels, self.n_sc, 1, 1)))#K^2,M,m,1,1
        self.aux_combination = nn.Parameter(init.kaiming_normal_(torch.rand(self.fre_num, self.out_channels, self.n_sc, 1, 1)))
        self.bais = nn.Parameter(nn.init.zeros_(torch.Tensor(out_channels))) if bais else None
    def forward(self, x):
        avg_freq = (self.fre_num * self.out_channels ) / self.n_sc
        self.proxy_combination=DiscreteFunction.apply(self.aux_combination, self.balance_weight, avg_freq, self.n_sc)
        mask = select_filter(self.n_sc, SCFConv2d.drop_filter_stage, self.last_rate, self.fre_num, self.sc)#[K^2,m,1,1,1]
        filter_3d=(self.sc).view(self.n_sc, self.basic_channels, 1, self.kernel_size, self.kernel_size)#(m,N,1,3,3)
        out=F.conv3d(x,filter_3d,padding=(0,self.kernel_size // 2,self.kernel_size // 2)) #output:B,m,K^2,W,H
        out=out*(mask.permute(4,1,0,2,3).contiguous())
        combination_filter=(self.aux_coefficients * self.proxy_combination).permute(1, 2, 0, 3, 4).contiguous() #(M,m,K^2,1,1)
        out=F.conv3d(out,combination_filter).squeeze(dim=2)#output:B,M,1,W,H->B,M,W,H
        if (not self.bais is None):
            out=out + self.bais.view(1, out.size(1),1,1).expand_as(out)
          

        return out

