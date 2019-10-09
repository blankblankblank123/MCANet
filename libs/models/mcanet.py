#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .resnet import _ConvBnReLU, _ResLayer, _Stem


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class MCANet_Infer(nn.Module):
    """
    Inference phase of MCANet
    """

    def __init__(self, n_blocks, atrous_rates):
        super(MCANet_Infer, self).__init__()

        self.encoder = nn.Sequential()
        ch = [64 * 2 ** p for p in range(6)]
        self.encoder.add_module("layer1", _Stem(ch[0]))
        self.encoder.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.encoder.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.encoder.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.encoder.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.FC_I = nn.Conv2d(ch[5],ch[5],kernel_size = 1)
        self.FC_P = nn.Conv2d(ch[5],ch[5],kernel_size = 1)

        self.base_conv = nn.Conv2d(ch[5],ch[5],kernel_size = 3,padding = 1)

        self.spatial_att_conv = nn.Sequential()
        self.spatial_att_conv.add_module('conv1',nn.Conv2d(ch[5],512,kernel_size = 3,padding = 1))
        self.spatial_att_conv.add_module('relu',nn.ReLU(inplace = True))
        self.spatial_att_conv.add_module('conv2',_ASPP(512, 1, atrous_rates))

        self.segmentation_module = nn.Sequential()
        self.segmentation_module.add_module('conv',nn.Conv2d(ch[5],512,kernel_size = 3, padding = 1))
        self.segmentation_module.add_module('relu',nn.ReLU(inplace = True))
        self.segmentation_module.add_module('aspp',_ASPP(512, 2, atrous_rates))


        self.relu = nn.ReLU(inplace = True)

    def initial_C_msg_passing(self,Input):
        F_list = []
        Fe_list = []
        CI_list = []
        for x in Input:
            x = self.encoder(x)
            Fe_list.append(x.cpu())
            x = self.relu(self.base_conv(x))
            x_ = self.gap(x)
            x_ = torch.sigmoid(self.FC_I(x_))
            F_list.append(x.cpu())
            CI_list.append(x_.cpu())

        CI = torch.cat(F_list,dim = 0)
        CIG = torch.mean(CI,dim = 0,keepdim = True).cuda()
        CIG = torch.sigmoid(self.FC_I(self.gap(CIG)))
        FC_list = []
        for x in F_list:
            x = x.cuda() * CIG
            FC_list.append(x.cpu())
        return FC_list,Fe_list,CI_list



    def S_msg_passing(self,FC_list,F_list):
        S_list = []
        b_list = []
        FCS_list = []
        for x_,x in zip(F_list,FC_list):
            S = self.spatial_att_conv(x.cuda())
            FCS = x_.cuda() * torch.sigmoid(S)
            FCS_list.append(FCS.cpu())
            S_list.append(S.cpu())
            b = self.gap(S)
            b = torch.sigmoid(b)
            b_list.append(b.cpu().view(1,1,1,1))

        return S_list,b_list,FCS_list

    def progressive_C_msg_passing_aggregation(self,FCS_list,F_list,b_list):
        FP_list = []
        CP_list = []
        for FCS in FCS_list:
            CP = self.gap(FCS)
            CP_list.append(CP.cpu())

        b_array = torch.cat(b_list,dim = 0)
        _,ind = torch.sort(b_array,dim = 0,descending = True)
        ind = ind.numpy()[:int(max(1,0.1 * (ind.shape[0])))].reshape(-1).tolist()

        CP_list = [CP_list[i] for i in ind]
        CP_array = torch.cat(CP_list,dim = 0)
        CPG = torch.mean(CP_array,dim = 0,keepdim = True)
        CPG = torch.sigmoid(self.FC_P(CPG.cuda())).cpu()
        return CPG



    def progressive_C_msg_passing_distribution(self,Input_set,CPG,scales):
        FP_list = []
        for x in Input_set:
            FP_pyramid = []
            for p in scales:
                h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
                f = self.encoder(h)
                FP = f * CPG.cuda()
                FP_pyramid.append(FP.cpu())
            FP_list.append(FP_pyramid)

        return FP_list

    def segmentation(self,FP):
        M = self.segmentation_module(FP.cuda())
        return M.cpu()

