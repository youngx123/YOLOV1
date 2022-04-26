# -*- coding: utf-8 -*-
# @Author : xyoung

import gc

import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt
import os
import imageio
import torch
from .augmentations import SSDAugmentation
from .data_transforms import Compose
from torchvision import transforms

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))


def Show_Bbox(image, box, size=1, center_format=False, ):
    if center_format:
        for b in box:
            pts1 = (int((b[0] - b[2]) * size), int((b[1] - b[3]) * size))
            pts2 = (int((b[0] + b[2]) * size), int((b[1] + b[3]) * size))
            cv2.rectangle(image, pts1, pts2, color=(255, 0, 0))
    else:
        for b in box:
            pts1 = (int(b[0] * size), int(b[1] * size))
            pts2 = (int(b[2] * size), int(b[3] * size))
            cv2.rectangle(image, pts1, pts2, color=(255, 0, 0))

    plt.imshow(image)
    plt.show()


class yoloDataset(Dataset):
    def __init__(self, txtname, imageSie=416, numBboxes=2, classNum=None,
                 train=True, scale=32, MaxObj=50):
        self.featureSize = imageSie // scale
        self.numBboxes = numBboxes
        self.classNum = classNum
        self.IMAGESIZE = imageSie
        self.train = train
        self.MaxObj = MaxObj
        self.ImgList, self.LabelList = self.readText(txtname)

        if self.classNum is None:
            self.classNum = len(VOC_CLASSES)

        if self.train:
            self.SSDAug = SSDAugmentation(self.IMAGESIZE, mean=(0, 0, 0), std=(1, 1, 1))

    def readText(self, path):
        self.ImgList = list()
        self.LabelList = list()
        with open(path, "r") as fid:
            labelInfo = fid.readlines()

        labelInfo = [line.strip() for line in labelInfo]
        for id, line in enumerate(labelInfo):
            line = line.strip()
            labelPath = line.replace("JPEGImages", "Label").replace("jpg", "txt")
            self.ImgList.append(line)
            self.LabelList.append(labelPath)
        return self.ImgList, self.LabelList

    def __len__(self):
        return len(self.ImgList)

    def encoder(self, bboxInfo, sizeW, sizeH):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return featuresize x featuresize x30
        '''
        grid_num = self.featureSize
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num

        for i in range(len(bboxInfo)):
            item = bboxInfo[i]
            cls_id = int(item[-1])
            # #bbox normalize to 0 ~ 1
            x1, y1 = item[0] / sizeW, item[1] / sizeH
            x2, y2 = item[2] / sizeW, item[3] / sizeH

            # # get center x,y cord (0~1 range)
            cxcy = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            wh = np.array([x2 - x1, y2 - y1])

            cxcy_sample = cxcy
            ij = np.ceil(cxcy_sample / cell_size) - 1  #

            # target [xoffset, yoffset, w,h, contain_obj, xoffset, yoffset, w,h, contain_obj, class_catlog]
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(cls_id) + 9] = 1
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = torch.from_numpy(wh)
            target[int(ij[1]), int(ij[0]), :2] = torch.from_numpy(delta_xy)
            target[int(ij[1]), int(ij[0]), 7:9] = torch.from_numpy(wh)
            target[int(ij[1]), int(ij[0]), 5:7] = torch.from_numpy(delta_xy)
        return target

    def encoder2(self, bboxInfo):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        target_tensor: (tensor) size(batchsize,S,S,30) predNum * (1+4) + classNum -> (1+4) + 20 = 25
        return target_tensor
        '''
        B = 2
        grid_num = self.featureSize

        target = torch.zeros((grid_num, grid_num, 30))
        boxes_wh = bboxInfo[:, 2:4] - bboxInfo[:, :2]  # width and height for each box, [n, 2]
        boxes_xy = (bboxInfo[:, :2] + bboxInfo[:, 2:4]) / 2.0  # center x & y for each box, [n, 2]
        labels = bboxInfo[:, -1]
        for i in range(len(bboxInfo)):
            xy, wh, label = boxes_xy[i], boxes_wh[i], int(labels[i])

            ij = (xy * grid_num)
            i, j = int(ij[0]), int(ij[1])  # y & x index which represents its location on the grid.

            xy_normalized = (ij - ij.astype(np.int))  # x & y of the box on the cell, normalized from 0.0 to 1.0.
            for k in range(B):
                s = 5 * k
                target[j, i, s:s + 2] = torch.from_numpy(xy_normalized)
                target[j, i, s + 2:s + 4] = torch.from_numpy(wh)
                target[j, i, s + 4] = 1.0
            target[j, i, 5 * B + label] = 1.0

        return target

    def __getitem__(self, item):
        imgPath = self.ImgList[item]
        labelPath = self.LabelList[item]

        if self.train:
            if os.path.exists(imgPath) and os.path.exists(labelPath):
                img = imageio.imread(imgPath)
                img = np.array(img, dtype=np.float)
                origin_H, origin_W = img.shape[:2]
                gt = np.loadtxt(labelPath, delimiter=",").reshape(-1, 5)
                gt[..., [0, 2]] = gt[..., [0, 2]] / origin_W
                gt[..., [1, 3]] = gt[..., [1, 3]] / origin_H
            else:
                raise "path don't exist, check again"

            img, box, label = self.SSDAug(img, gt[..., :4], gt[..., -1])
            # Show_Bbox(img, box, self.IMAGESIZE)
            img = img.transpose(2, 0, 1)
            gt = np.concatenate((box, label.reshape(-1, 1)), 1)

            gt[..., [0, 2]] = gt[..., [0, 2]]  # 0~1  # * self.IMAGESIZE
            gt[..., [1, 3]] = gt[..., [1, 3]]  # 0~1  # * self.IMAGESIZE
            # # gt encode
            # gt = self.encoder2(gt)

            num = 5
            filledGt = np.zeros((self.MaxObj, num), np.float32)
            filledGt[range(len(gt))[:self.MaxObj]] = gt[:self.MaxObj]
            return torch.from_numpy(img), torch.from_numpy(filledGt)
        else:
            img = imageio.imread(imgPath)
            img = np.array(img, dtype=np.float)
            img = img / 255.0
            img = cv2.resize(img, (self.IMAGESIZE, self.IMAGESIZE))
            img = img.transpose(2, 0, 1)
            return torch.from_numpy(img), imgPath


if __name__ == '__main__':
    dirPath = "voc2007.txt"
    txtName = r"D:\MyNAS\SynologyDrive\Object_Detection\v1\yolov1\2007train.txt"
    dataset = yoloDataset(txtName, classNum=20, train=True)

    dataloader = DataLoader(dataset, batch_size=2)
    for e in range(50):
        #     # progressBar = tqdm(dataloader)
        for batch in dataloader:
            pass
