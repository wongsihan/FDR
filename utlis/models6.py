#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm


class CNNEncoder2d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNNEncoder2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool2d((5, 5))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork2d(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size * 2, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class CNNEncoder1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNNEncoder1d, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
        #     nn.BatchNorm1d(64, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2))
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3)
        self.conv1v1 = nn.Conv1d(1, 64, kernel_size=13, padding=6, stride=1)
        #self.conv1v2 = nn.Conv1d(1, 64, kernel_size=3, padding=1, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Conv1d(64, 64 // 64, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(64 // 64, 64, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()


        self.IN1 =  nn.InstanceNorm1d(64, affine=True)
        self.BN1 =  nn.BatchNorm1d(64, momentum=1, affine=True)
        self.ac1 =  nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        # self.layer2 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=3, padding=0),
        #     nn.BatchNorm1d(64, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2))
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=0)
        self.IN2 = nn.InstanceNorm1d(64, affine=True)
        self.BN2 = nn.BatchNorm1d(64, momentum=1, affine=True)
        self.ac2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)


        # self.layer3 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(64, momentum=1, affine=True),
        #     nn.ReLU())
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        self.BN3 = nn.BatchNorm1d(64, momentum=1, affine=True)
        self.IN3 = nn.InstanceNorm1d(64, affine=True)
        self.ac3 = nn.ReLU()


        # self.layer4 = nn.Sequential(
        #     nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
        #     nn.ReLU())

        self.conv4 = nn.Conv1d(64, feature_dim, kernel_size=3, padding=1)
        self.BN4 = nn.BatchNorm1d(feature_dim, momentum=1, affine=True)
        self.IN4 = nn.InstanceNorm1d(feature_dim, affine=True)
        self.ac4 = nn.ReLU()


        # self.layer5 = nn.Sequential(
        #     nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
        #     nn.ReLU())
        # self.layer6 = nn.Sequential(
        #     nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
        #     nn.ReLU())
        # self.layer7 = nn.Sequential(
        #     nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
        #     nn.ReLU())
        # self.layer8 = nn.Sequential(
        #     nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
        #     nn.ReLU())
        # self.layer9 = nn.Sequential(
        #     nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
        #     nn.ReLU())
        # self.layer10 = nn.Sequential(
        #     nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
        #     nn.ReLU())


        self.admp = nn.AdaptiveMaxPool1d(25)

    def forward(self, x):
        per_out = []

        #out = self.conv1(x)

        outv1 = self.conv1v1(x)
        # outv1_mean = torch.mean(outv1,dim=2).unsqueeze(2).expand(x.size(0),64,1024)
        # sub_out = outv1 - outv1_mean
        #
        # r = x - outv1
        # r_mean = torch.mean(r, dim=2).unsqueeze(2).expand(x.size(0), 64, 1024)
        # sub_r = r - r_mean
        #
        # sim1 = torch.cosine_similarity(sub_out,sub_r,2)
        # # sim2 = torch.cosine_similarity(outv2, r, 2)
        # att_mat1 = torch.softmax(sim1,dim=1)
        #
        # # att_mat2 = torch.softmax(sim2, dim=1)
        # att_mat1 = att_mat1.unsqueeze(2).expand(x.size(0), 64, 1024)
        # r_plus = torch.mul(att_mat1, r)
        # out = outv1 + r_plus
        # #out = torch.mul(att_mat1, out)

        # r1 = x - outv1
        # r = self.avg_pool(r1)
        # r = self.fc1(r)
        # r = self.relu(r)
        # r = self.fc2(r)
        # a_vec = self.sigmoid(r).expand(x.size(0), 64, 1024)
        # r_plus = torch.mul(a_vec, r1)
        # out = outv1 + r_plus

        r = x - outv1
        r = self.avg_pool(r)
        r = self.fc1(r)
        r = self.relu(r)
        r = self.fc2(r)
        r = self.sigmoid(r)
        a_vec = self.sigmoid(r).expand(x.size(0), 64, 1024)
        r_plus = torch.mul(a_vec, r)
        out = outv1 + r_plus

        out = self.BN1(out)
        out = self.ac1(out)
        out = self.pool1(out)

       # out = out + identity
       # out = self.layer1(x)
        per_out.append(out)
        out = self.conv2(out)
        # split = torch.split(out, 32, 1)
        # out3 = self.IN2(split[0].contiguous())
        # out4 = self.BN2(split[1].contiguous())
        # out = torch.cat((out3, out4), 1)
        out = self.BN2(out)
        out = self.ac2(out)
        out = self.pool2(out)
        #out = self.layer2(out)
        per_out.append(out)


        out = self.conv3(out)
        #
        # r2 = out - outv2
        # r2 = self.avg_pool2(r2)
        # r2 = self.fc12(r2)
        # r2 = self.relu2(r2)
        # r2 = self.fc22(r2)
        # r2 = self.sigmoid2(r2)
        # a_vec2 = self.sigmoid(r2).expand(x.size(0), 64, 255)
        # r_plus2 = torch.mul(a_vec2, r2)
        # out = outv2 + r_plus2
        out = self.BN3(out)
        #out = self.IN3(out)
        out = self.ac3(out)
        per_out.append(out)
        #out = self.layer4(out)
        out = self.conv4(out)
        out = self.BN4(out)


        out = self.ac4(out)
        per_out.append(out)
        # out = self.layer5(out)
        # per_out.append(out)
        # out = self.layer6(out)
        # per_out.append(out)
        # out = self.layer7(out)
        # per_out.append(out)
        # out = self.layer8(out)
        # per_out.append(out)
        # out = self.layer9(out)
        # per_out.append(out)
        # out = self.layer10(out)
        # per_out.append(out)
        out = self.admp(out)

        # out = out.view(out.size(0),-1)
        return out,per_out  # 64


class RelationNetwork1d(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        num = int(input_size / 2)
        super(RelationNetwork1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size * 2, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))



        self.layer2 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        # self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.sfc1 = nn.Conv1d(input_size, input_size // 16, kernel_size=1,
        #                       padding=0)
        # self.relu = nn.ReLU(inplace=True)
        # self.sfc2 = nn.Conv1d(input_size // 16, input_size, kernel_size=1,
        #                       padding=0)
        # self.sigmoid = nn.Sigmoid()
        # self.BN1 = nn.BatchNorm1d(input_size, momentum=1, affine=True)
        # self.ac1 = nn.ReLU()
        # self.pool1 = nn.MaxPool1d(2)
        #
        self.fc1 = nn.Linear(input_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)

        # out = self.conv1(out1)
        # r1 = out1 - out
        # # print("r", r.size()) 15 64 25
        # r = self.avg_pool(r1)
        # r = self.sfc1(r)
        # r = self.relu(r)
        # r = self.sfc2(r)
        # # print("r", r.size())15 64 1
        # a_vec = self.sigmoid(r).expand(x.size(0), 64, 12)
        # # print("a_vec", a_vec.size()) 15 128 1024
        # r_plus = torch.mul(r1, a_vec)
        # out = out + r_plus
        # out = self.BN1(out)
        # out = self.ac1(out)
        # out = self.pool1(out)

        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class Classifier1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, input_size, out_size):
        super(Classifier1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc1 = nn.Linear(input_size * 6, out_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class CNN1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNN1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool1d(25)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        out = out.view(out.size(0),-1)
        return out


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores