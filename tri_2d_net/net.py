# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import torch
import torch.nn as nn
import torchvision.models as models


class AttBranch(nn.Module):
    def __init__(self):
        super(AttBranch, self).__init__()
        _net = models.vgg11_bn()
        _net_list = list(_net.children())[0]
        self.backbone2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, dilation=2, padding=2, bias=False),
            *_net_list[1:-14],
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1))

        nn.init.constant_(self.backbone2d[-1].weight, 0)
        nn.init.constant_(self.backbone2d[-1].bias, 0)

    def forward(self, x):
        x = self.backbone2d(x)
        return x.clamp(min=0)


class Branch(nn.Module):
    def __init__(self, num_classes=2, dout=False):
        super(Branch, self).__init__()
        self.att_branch = AttBranch()
        _net = models.resnet18()
        _net_list = list(_net.children())
        self.backbone2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, dilation=2, padding=2, bias=False),
            *_net_list[1:-3])
        self.aux = nn.Sequential(
            *_net_list[-3:-1],
            nn.Flatten())
        self.fc = nn.Linear(512, num_classes)
        self.dout = None
        if dout:
            self.dout = nn.Dropout()

    def forward(self, x):
        n, d, c, h, w = x.size()
        x_org = x.view(n * d, c, 1, h, w).contiguous()
        x = self.backbone2d(x_org[:, 0, :, :, :])
        att = self.att_branch(x_org[:, 1, :, :, :])
        x = x * (att + 1)
        _, c, h, w = x.size()
        x = x.view(n, d, c, h, w)
        # -> n, c, d, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        aux_feature = x.max(dim=2)[0]
        aux_feature = self.aux(aux_feature)
        aux_feature = aux_feature / aux_feature.norm(dim=1, keepdim=True)
        if self.dout is not None:
            aux_feature = self.dout(aux_feature)
        aux_pred = self.fc(aux_feature)

        return aux_pred, aux_feature


class Tri2DNet(nn.Module):
    def __init__(self, num_classes=2, dout=False):
        super(Tri2DNet, self).__init__()
        self.num_classes = num_classes
        self.branch_axial = Branch(num_classes, dout)
        self.branch_sagittal = Branch(num_classes, dout)
        self.branch_coronal = Branch(num_classes, dout)
        self.fc_fuse = nn.Linear(512 * 3, num_classes)
        self.dout = None
        if dout:
            self.dout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, d, h, w = x.size()
        # -> n, w, c, h, d
        x_sagittal = x.permute(0, 4, 1, 3, 2).contiguous()
        # -> n, h, c, d, w
        x_coronal = x.permute(0, 3, 1, 2, 4).contiguous()
        # -> n, d, c, h, w
        x_axial = x.permute(0, 2, 1, 3, 4).contiguous()
        del x
        aux_pred_sagittal, aux_feature_sagittal = self.branch_sagittal(x_sagittal)
        aux_pred_coronal, aux_feature_coronal = self.branch_coronal(x_coronal)
        aux_pred_axial, aux_feature_axial = self.branch_axial(x_axial)
        feature = torch.cat([aux_feature_sagittal, aux_feature_coronal, aux_feature_axial], dim=1)
        feature = feature / feature.norm(dim=1, keepdim=True)
        pred = self.fc_fuse(feature)

        return pred, aux_pred_sagittal, aux_pred_coronal, aux_pred_axial
