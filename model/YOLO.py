# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 17:18  2021-07-07
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from .darknet import darknet53, darknet19
from .HEAD import V3Head, VHhead, YOLOXHead


class YOLOV1(nn.Module):
    def __init__(self, class_num, v1head=False, v3Head=False):
        super(YOLOV1, self).__init__()
        self.backbone = darknet53()
        self.scale = 32
        self.out_filters = self.backbone.layers_out_filters
        if v1head:
            self.head = VHhead(self.out_filters[-1], class_num)
        elif v3Head:
            self.head = V3Head(self.out_filters[-1], [self.out_filters[-2], self.out_filters[-1]], class_num)
        else:
            self.head = YOLOXHead(class_num, [self.out_filters[-1]])

    def forward(self, x):
        # # [batch_size, {256,512, 1024}, fsize,fsize ], fsize =[13, 26, 52]
        _, _, x5 = self.backbone(x)

        # pred
        out5 = self.head(x5)

        if isinstance(out5, list):
            out5 = out5[0]
        return out5


if __name__ == '__main__':
    yolov1 = YOLOV1(20)
    a = torch.randn(1, 3, 448, 448)
    out = yolov1(a)
    for i in out:
        print(out.shape)
