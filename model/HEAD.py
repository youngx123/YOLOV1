# -*- coding: utf-8 -*-
# @Author : youngx

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import OrderedDict
import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if leaky else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class SPP(nn.Module):
    def __init__(self, pool_sizes=[1, 5, 9, 13]):
        super(SPP, self).__init__()
        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes]
        )

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features, dim=1)
        return features

# Copy from yolov5
class Bottleneck2(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck2, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=3, p=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)
        self.cv4 = Conv(2 * c_, c2, k=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck2(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VHhead(nn.Module):
    def __init__(self, in_fileter, classNum, predNum=2):
        super(VHhead, self).__init__()
        self.classNum = classNum
        self.predNum = predNum
        self.spp = SPP()
        self.conv1x1 = Conv(in_fileter*4, in_fileter, k=1)
        self.conv = BottleneckCSP(in_fileter, in_fileter//2, n=3, shortcut=False)
        self.pred = nn.Conv2d(in_fileter//2, self.predNum * (1 + 4) + self.classNum, kernel_size=1)

    def forward(self, x):
        x = self.spp(x)
        x = self.conv1x1(x)
        x = self.conv(x)
        x = self.pred(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return x


#                       #
#  yolox detection head #
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, in_channels=[1024], width=1.0,  act="silu", depthwise=False, ):
        super().__init__()
        Conv = BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()  # 两个3x3的卷积

        self.cls_preds = nn.ModuleList()  # 一个1x1的卷积，把通道数变成类别数，比如coco 80类
        self.reg_preds = nn.ModuleList()  # 一个1x1的卷积，把通道数变成4通道，因为位置是xywh
        self.obj_preds = nn.ModuleList()  # 一个1x1的卷积，把通道数变成1通道，判断有无目标
        self.stems = nn.ModuleList()  # 模前面的 BaseConv模块

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width),
                         ksize=1, stride=1, act=act))

            self.cls_convs.append(
                nn.Sequential(
                    *[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                      Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                      ])
            )
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )

            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)  # 3
            )

    def forward(self, inputs):
        outputs = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # get predict class category result
            cls_output = self.cls_preds[k](cls_feat)

            # get xywh predict result
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)

            #  get if contain object probability
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output.permute(0, 2, 3, 1))
        return outputs


#                    #
#  v3 detection head #
#                    #
def conv2d(c_in, c_out, kernel):
    '''
    cbl = conv + batch_norm + leaky_relu
    '''
    pad = (kernel - 1) // 2 if kernel else 0
    conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=1, padding=pad, bias=False)
    b1 = nn.BatchNorm2d(c_out)
    relu1 = nn.LeakyReLU(0.1)
    return nn.Sequential(OrderedDict([
        ("conv", conv1),
        ("bn", b1),
        ("relu", relu1),
    ]))


class V3Head(nn.Module):
    def __init__(self, in_filters, filters_list, classNum):
        super(V3Head, self).__init__()
        self.classNum = classNum
        self.predNum = classNum
        self.conv5times = self.makeFiveLayer(in_filters, filters_list)
        self.pred = nn.Sequential(
            conv2d(filters_list[0], filters_list[1], 3),
            nn.Conv2d(filters_list[1], 3*(1 + 4 + self.classNum), kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.conv5times(x)
        x = self.pred(x)
        x = torch.sigmoid(x)
        return x.permute(0, 2, 3, 1)

    def makeFiveLayer(self, in_filters, filters_list):
        layer = nn.Sequential(
            conv2d(in_filters, filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1)
        )
        return layer


if __name__ == '__main__':
    a5 = torch.randn((1, 1024, 13, 13))
    a4 = torch.randn((1, 512, 26, 26))
    a3 = torch.randn((1, 256, 52, 52))

    net = VHhead(512, 20)
    out = net(a4)
    for a in out:
        print(a.shape)
