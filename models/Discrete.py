# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.autograd import Function

class DiscreteFunction(Function):

    @staticmethod
    def forward(ctx, aux_combination, balance_weight, avg_freq, n_lego):
        balance_weight=torch.FloatTensor([balance_weight]).to(aux_combination.device)
        avg_freq=torch.FloatTensor([avg_freq]).to(aux_combination.device)
        n_lego=torch.LongTensor([n_lego]).to(aux_combination.device)
        ctx.save_for_backward(aux_combination, balance_weight, avg_freq, n_lego)
        proxy_combination = torch.zeros(aux_combination.size()).to(aux_combination.device)
        proxy_combination.scatter_(2, aux_combination.argmax(dim = 2, keepdim = True), 1)
        proxy_combination.requires_grad = True
        return proxy_combination

    @staticmethod
    def backward(ctx, grad_output):
        aux_combination = ctx.saved_tensors[0]
        balance_weight = ctx.saved_tensors[1]
        avg_freq= ctx.saved_tensors[2]
        n_lego =int(ctx.saved_tensors[3].view(-1).cpu().numpy())
        balance_weight=float(balance_weight.view(-1).cpu().numpy())
        avg_freq=float(avg_freq.view(-1).cpu().numpy())
        # balance loss
        idxs = aux_combination.argmax(dim = 2).view(-1).cpu().numpy()
        unique, count = np.unique(idxs, return_counts = True)
        unique, count = np.unique(count, return_counts = True)
        #avg_freq = (self.fre_num * self.out_channels ) / self.n_lego
        max_freq = 0
        min_freq = 100
        for i in range(n_lego):
            i_freq = (idxs == i).sum().item()
            max_freq = max(max_freq, i_freq)
            min_freq = min(min_freq, i_freq)
            if i_freq >= np.floor(avg_freq) and i_freq <= np.ceil(avg_freq):
                continue
            if i_freq < np.floor(avg_freq):
                s = float(balance_weight * (np.floor(avg_freq) - i_freq))
                grad_output[:, :, i] = grad_output[:, :, i] - s
                
            if i_freq > np.ceil(avg_freq):
                s = float(balance_weight * (i_freq - np.ceil(avg_freq)))
                grad_output[:, :, i] = grad_output[:, :, i] + s

        return grad_output, None, None, None
            
