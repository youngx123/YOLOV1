# -*- coding: utf-8 -*-
# @Author : xyoung

import gc
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import OrderedDict
import torch
import torch.nn as nn


class SAM(nn.Module):
    """ Parallel CBAM """

    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Spatial Attention Module """
        x_attention = self.conv(x)

        return x * x_attention


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """

    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 3, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 7, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


class ConvBNLeakyReLU(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBNLeakyReLU, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(c_out[0], c_out[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = out + residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self.make_layer_([32, 64], layers[0])
        self.layer2 = self.make_layer_([64, 128], layers[1])
        self.layer3 = self.make_layer_([128, 256], layers[2])
        self.layer4 = self.make_layer_([256, 512], layers[3])
        self.layer5 = self.make_layer_([512, 1024], layers[4])
        self.layers_out_filters = [64, 128, 256, 512, 1024]

    def make_layer_(self, filters, num_block):
        layers = []
        # maxpooling layer
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, filters[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(filters[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # res block
        self.inplanes = filters[1]
        for i in range(0, num_block):
            layers.append(("residual_{}".format(i), ConvBNLeakyReLU(self.inplanes, filters)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x3, x4, x5


def darknet53(pretrained=None):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        pretrained = "./weights/darknet53_backbone_weights.pth"
        pretrain_weight = torch.load(pretrained)
        key = pretrain_weight.keys()
        modelDict = model.state_dict()
        for k in modelDict.keys():
            if k in key and modelDict[k].shape == pretrain_weight[k].shape:
                modelDict[k] = pretrain_weight[k]
            else:
                print("skip layer : ", k)
        model.load_state_dict(modelDict)
        del pretrain_weight, modelDict
        gc.collect()
        torch.cuda.empty_cache()
        print("loade darknet53 pretrained model")
    return model


def darknet19(pretrained=None):
    model = DarkNet([1, 3, 3, 5, 5])
    if pretrained:
        pretrained = "./weights/darknet19_backbone_weights.pth"
        pretrain_weight = torch.load(pretrained)
        key = pretrain_weight.keys()
        modelDict = model.state_dict()
        for k in modelDict.keys():
            if k in key and modelDict[k].shape == pretrain_weight[k].shape:
                modelDict[k] = pretrain_weight[k]
            else:
                print("skip layer : ", k)
        model.load_state_dict(modelDict)
        del pretrain_weight, modelDict
        gc.collect()
        torch.cuda.empty_cache()
        print("loade darknet19 pretrained model")
    return model


if __name__ == '__main__':
    model = darknet53(True)
    model_dict = model.state_dict()

    print('Finished!')
