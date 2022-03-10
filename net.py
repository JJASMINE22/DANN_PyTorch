# -*- coding: UTF-8 -*-
'''
@Project ：DANN
@File    ：dann_net.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
from torch import nn


class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.leak1 = nn.LeakyReLU(negative_slope=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.leak2 = nn.LeakyReLU(negative_slope=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.weights = self.init_params()

    def forward(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.leak1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leak2(x)
        output = self.pool2(x)

        return output

    def init_params(self):
        """
        store weights, for regularization
        """
        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'bias':
                    torch.nn.init.zeros_(param)
                else:
                    stddev = 1/math.sqrt(param.shape[0])
                    torch.nn.init.normal_(param, std=stddev)
                    weights.append(param)

        return weights

class Domain_classifier(nn.Module):
    def __init__(self):
        super(Domain_classifier, self).__init__()

        self.conv = nn.Conv2d(in_channels=128, out_channels=64,
                              kernel_size=3)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.leak = nn.LeakyReLU(negative_slope=0.2)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.linear = nn.Linear(in_features=64, out_features=1)

        self.weights = self.init_params()

    def forward(self, input):

        x = self.conv(input)
        x = self.bn(x)
        x = self.leak(x)

        x = self.max_pool(x).view(-1, 64)
        output = self.linear(x)

        return output

    def init_params(self):
        """
        store weights, for regularization
        """
        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'bias':
                    torch.nn.init.zeros_(param)
                else:
                    stddev = 1/math.sqrt(param.shape[0])
                    torch.nn.init.normal_(param, std=stddev)
                    weights.append(param)
        return weights

class Label_predictor(nn.Module):
    def __init__(self):
        super(Label_predictor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.leak1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64,
                               kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.leak2 = nn.LeakyReLU(negative_slope=0.2)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.linear = nn.Linear(in_features=64, out_features=10)

        self.weights = self.init_params()

    def forward(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.leak1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leak2(x)

        x = self.max_pool(x).view(-1, 64)
        x = self.linear(x)

        output = torch.sigmoid(x)

        return output

    def init_params(self):
        """
        store weights, for regularization
        """
        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'bias':
                    torch.nn.init.zeros_(param)
                else:
                    stddev = 1/math.sqrt(param.shape[0])
                    torch.nn.init.normal_(param, std=stddev)
                    weights.append(param)

        return weights
