"""
    Definition of Compund_DCT Residual Networks.

    Licensed under the BSD License [see LICENSE for details].

    Written by Yao Lu, based on torchvision implementation:
    https://github.com/pytorch/vision/tree/master/torchvision/models
"""

import torch.nn as nn
import torch.nn.functional as F
#from utils import load_pretrained
from models.Compund_DCT import CompundDCT_Conv
from models.SCF import SCFConv2d
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnet_cifar_56', 'resnet_cifar_110']


def CompundDCT_Conv3x3(in_planes, out_planes, stride=1, compund_level=1, level=None, groups=2, last_rate=1., balance_weight=1e-4):
    """3x3 harmonic convolution with padding"""
    return CompundDCT_Conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                               bias=False, use_bn=True,  compund_level=compund_level, level=level, groups=groups, last_rate=last_rate,  balance_weight= balance_weight)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class WeightedPool(nn.Module):#non-overlap-pooling
    def __init__(self, planes, kernel_size, padding=0):
        super(WeightedPool,self).__init__()
        self.planes=planes
        self.kernel_size=kernel_size
        self.padding=padding
        self.pool_weight=nn.Parameter((torch.ones(1,1,kernel_size,kernel_size)/(kernel_size**2)), requires_grad=True)
    def forward(self,x):
        filt=self.pool_weight.expand(self.planes,1,self.kernel_size,self.kernel_size).contiguous()
        out=F.conv2d(x, filt, stride=self.kernel_size, padding=self.padding,  groups=self.planes)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, DCT_flag=True, compund_level=1, level=None, groups=1, last_rate=1., balance_weight=1e-4):
        super(BasicBlock, self).__init__()
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if DCT_flag:
            #print(last_rate)
            self.CompundDCT_Conv1 = CompundDCT_Conv3x3(inplanes, planes, stride, compund_level=compund_level, level=level, groups=groups, last_rate=last_rate,  balance_weight= balance_weight)
            self.CompundDCT_Conv2 = CompundDCT_Conv3x3(planes, planes, compund_level=compund_level, level=level, groups=groups, last_rate=last_rate,  balance_weight= balance_weight)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.CompundDCT_Conv1(x) if hasattr(self, 'CompundDCT_Conv1') else self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.CompundDCT_Conv2(out) if hasattr(self, 'CompundDCT_Conv2') else self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, DCT_flag=True, compund_level=1, level=None, groups=1, last_rate=1.,  balance_weight=1e-4):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        #self.conv1 = conv1x1(inplanes, planes)
        self.conv1 = SCFConv2d( in_channels=inplanes, out_channels=planes,bais=False, kernel_size=1,  n_lego=0.5, fre_num=2, last_rate=1.,balance_weight= balance_weight)
        self.bn1 = nn.BatchNorm2d(planes)
        if DCT_flag:
            self.CompundDCT_Conv2 = CompundDCT_Conv3x3(planes, planes, stride, compund_level=compund_level, level=level, groups=groups, last_rate=last_rate,  balance_weight= balance_weight)
        else:
            self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = conv1x1(planes, planes * self.expansion)
        self.conv3 = SCFConv2d( in_channels=planes, out_channels=planes* self.expansion,bais=False, kernel_size=1,  n_lego=0.5, fre_num=2, last_rate=1.,balance_weight= balance_weight)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.CompundDCT_Conv2(out) if hasattr(self, 'CompundDCT_Conv2') else self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, dataset, block=BasicBlock, layers=[2,2,2,2], num_classes=1000, DCT_root=True, DCT_flag=True, pool=None,compund_level=1., levels=[None, None, None, None], last_rates=[1.,1.,1.,1.],groups_list=[2,2,2,2], depth=56, bottleneck=True,  balance_weight= 1e-5):
        super(ResNet, self).__init__()
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock
            if DCT_root:
                self.CompundDCT_firstConv1 = CompundDCT_Conv(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, use_bn=True,  compund_level=1, level=None, groups=2, last_rate=1.,  balance_weight= balance_weight)
            else:
                self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n, DCT_flag=DCT_flag, compund_level=compund_level, level=levels[0], groups=groups_list[0], last_rate=last_rates[0],  balance_weight= balance_weight)
            self.layer2 = self._make_layer(block, 32, n, stride=2, DCT_flag=DCT_flag, compund_level=compund_level, level=levels[1], groups=groups_list[1], last_rate=last_rates[1],  balance_weight= balance_weight)
            self.layer3 = self._make_layer(block, 64, n, stride=2, DCT_flag=DCT_flag, compund_level=compund_level, level=levels[2], groups=groups_list[2], last_rate=last_rates[2],  balance_weight= balance_weight)
            self.avgpool = nn.AvgPool2d(8)
            #self.avgpool=WeightedPool(planes=(64 * block.expansion), kernel_size=8, padding=0)
            self.fc = nn.Linear(64 * block.expansion, num_classes)
            
        else:
            self.inplanes = 64
            
            root_stride = 2 if pool in ['avg', 'max'] else 4
            if DCT_root:
                self.CompundDCT_firstConv1 = CompundDCT_Conv(3, 64, kernel_size=7, stride=roor_stride, padding=3, bias=False, use_bn=True,  compund_level=1, level=None, groups=2, last_rate=1.,  balance_weight= balance_weight)
            else:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=root_stride, padding=3, bias=False)
            
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            if pool == 'avg':
                self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            elif pool == 'max':
                self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0], DCT_flag=DCT_flag, compund_level=compund_level, level=levels[0], groups=groups_list[0], last_rate=last_rates[0],  balance_weight= balance_weight)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, DCT_flag=DCT_flag,compund_level=compund_level, level=levels[1], groups=groups_list[1], last_rate=last_rates[1],  balance_weight= balance_weight)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, DCT_flag=DCT_flag, compund_level=compund_level, level=levels[2], groups=groups_list[2], last_rate=last_rates[2],  balance_weight= balance_weight)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, DCT_flag=DCT_flag, compund_level=compund_level, level=levels[3], groups=groups_list[3], last_rate=last_rates[3],  balance_weight= balance_weight)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, DCT_flag=True, compund_level=1, level=None, groups=1, last_rate=1.,  balance_weight=1e-4):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                WeightedPool(planes=self.inplanes, kernel_size=2, padding=0),
                conv1x1(self.inplanes, planes * block.expansion, 1),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, DCT_flag, compund_level, level, groups, last_rate,  balance_weight))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None,  DCT_flag, compund_level, level, groups, last_rate,  balance_weight))

        return nn.Sequential(*layers)

    def forward(self, x, drop_filter_stage):
        CompundDCT_Conv.drop_filter_stage=drop_filter_stage
        #CompundDCT_Conv.weight_list=weight_list
        if self.dataset.startswith('cifar'):
            x = self.CompundDCT_firstConv1(x) if hasattr(self, 'CompundDCT_firstConv1') else self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x_maxpool = F.max_pool2d(x,x.size(2)).view(x.size(0),-1)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1) + 0.1*x_maxpool
            x = self.fc(x)
        else:
            x = self.CompundDCT_firstConv1(x) if hasattr(self, 'CompundDCT_firstConv1') else self.conv1(x) #112
            x = self.bn1(x)
            x = self.relu(x)
            if hasattr(self, 'pool'):
                x = self.pool(x) #56

            x = self.layer1(x) #56
            x = self.layer2(x) #28
            x = self.layer3(x) #14
            x = self.layer4(x) #7

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

    def copy_grad(self, balance_weight):
        for layer in self.modules():
            if isinstance(layer, SCFConv2d):
                layer.copy_grad(balance_weight)

def resnet18(pretrained=False, **kwargs):
    model = ResNet('', BasicBlock,  [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet('', BasicBlock,  [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet('', Bottleneck, [3, 4, 6, 3], **kwargs)
    #if pretrained and kwargs['harm_root'] and kwargs['harm_res_blocks'] and (not 'pool' in kwargs or not kwargs['pool'] in ['avg', 'max']) \
    #and (not 'levels' in kwargs or kwargs['levels'] == [None, None, None, None]):
    #    load_pretrained(model, 'https://github.com/matej-ulicny/harmonic-networks/releases/download/0.1.0/harm_resnet50-eec30392.pth')
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet('', Bottleneck, [3, 4, 23, 3], **kwargs)
    #if pretrained and kwargs['harm_root'] and kwargs['harm_res_blocks'] and (not 'pool' in kwargs or not kwargs['pool'] in ['avg', 'max']) \
    #and (not 'levels' in kwargs or kwargs['levels'] == [None, None, None, None]):
    #    load_pretrained(model, 'https://github.com/matej-ulicny/harmonic-networks/releases/download/0.1.0/harm_resnet101-62e185b1.pth')
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet('', Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet_cifar_56(depth=56, **kwargs):
    model = ResNet('cifar', depth=depth, **kwargs)
    return model

def resnet_cifar_110(depth=110, **kwargs):
    model = ResNet('cifar', depth=depth, **kwargs)
    return model
