

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=10)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    print(len(dataset))
    for inputs, targets in dataloader:
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)

mean_cifar100 = (0.5071, 0.4866, 0.4409)
std_cifar100 = (0.2009, 0.1984, 0.2023)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / float(self.count)
             
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
if __name__ == '__main__':
    # Mean and std used to normalize cifar10.
    transform_train = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    mean, std = get_mean_and_std(trainset)
    print(mean) # output: (0.4914, 0.4822, 0.4465)
    print(std)  # output: (0.2023, 0.1994, 0.2010)

    # Mean and std used to normalize cifar100.
    transform_train = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    mean, std = get_mean_and_std(trainset)
    print(mean) # output: (0.5071, 0.4866, 0.4409)
    print(std)  # output: (0.2009, 0.1984, 0.2023)



