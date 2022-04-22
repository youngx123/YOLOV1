# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 14:59  2021-07-12
import torch
import torch.nn as nn
import torch.nn.functional as F


class yoloLoss(nn.Module):
    def __init__(self, l_coord=5, l_noobj=0.5):
        super(yoloLoss, self).__init__()
        self.lambda_coord = l_coord
        self.lambda_noobj = l_noobj
        self.conf_loss = nn.BCELoss()
        self.cls_loss = nn.BCELoss()
        self.MSE_loss = nn.MSELoss()
        self.twh_loss = nn.MSELoss()

    def forward(self, pred, target):
        """
        :param pred:  size(batchsize,S,S,1x5+20=25) [x,y,w,h,c]
        :param target: size(batchsize,MaxObjNum, 5)
        :return:
        """
        BS = pred.shape[0]
        gridSize = pred.shape[2]

        pred_cxywh = pred[..., :10]  # (bs, s, s, 10)
        pred_cls = pred[..., 10:]  # (bs, s, s, 20)
        del pred
        torch.cuda.empty_cache()

        # # get target
        target_confxywh, target_ClC = self.build_target(target, pred_cxywh, gridSize)

        # # loss cal
        objMask = (target_confxywh[:, 0, :, :, 0] + target_confxywh[:, 1, :, :, 0]) == 1  # [n_batch, S, S]
        noobjMask = (target_confxywh[:, 0, :, :, 0] + target_confxywh[:, 1, :, :, 0]) == 0

        pred_cxywh = pred_cxywh.view(BS, gridSize, gridSize, 2, 5).permute(0, 3, 1, 2, 4)
        # no obj conf loss
        noobjMask = noobjMask.unsqueeze(1).repeat(1, 2, 1, 1)
        noobjPred = pred_cxywh[noobjMask]
        noobjTarget = target_confxywh[noobjMask]

        noobj_loss = F.mse_loss(noobjPred[..., 0], noobjTarget[..., 0])

        # # obj pred and target
        objclcPred = pred_cls[objMask]
        objclcTarget = target_ClC[objMask]

        objMask = objMask.unsqueeze(1).repeat(1, 2, 1, 1)
        objPred = pred_cxywh[objMask]
        objTarget = target_confxywh[objMask]

        # # cause target_confxywh and objMask shape is [B, 2, grid,grid, 5]
        # # means if objMask[B, 0,...] = 1 then [B, 0,...] is 0
        # # so the following method gets the true label
        selectMask = [False if objt.sum() == 0 else True for objt in objTarget]
        objTarget = objTarget[selectMask]
        objPred = objPred[selectMask]

        # obj loss
        objconf_loss = F.mse_loss(objPred[..., 0], objTarget[..., 0])
        loss_xy = F.mse_loss(objPred[..., 1:3], objTarget[..., 1:3])
        loss_wh = F.mse_loss(torch.sqrt(objPred[..., 3:5]), torch.sqrt(objTarget[..., 3:5]))

        loss_clc = F.mse_loss(objclcPred, objclcTarget)

        loss = self.lambda_coord * (loss_xy + loss_wh) + objconf_loss + self.lambda_noobj * noobj_loss + loss_clc
        return loss

    def build_target(self, target, predcxywh, grids):
        """
        Args:
        target: x1,x2,y1,y2 label , (x1,y1,x2,y2) 0~1
        predcxywh:
        grids: feature map size
        Returns:
        """
        bs = target.size(0)
        # # (bs, 2, grids, grids, 5) [conf, x,y,w,h]
        predcxywh = predcxywh.view(bs, grids, grids, 2, 5).permute(0, 3, 1, 2, 4)
        confxywh = torch.zeros((bs, 2, grids, grids, 5)).to(target.device)
        clcgt = torch.zeros(bs, grids, grids, 20).to(target.device)

        for bszie in range(bs):
            tg = target[bszie]
            for idx, gtItem in enumerate(tg):
                if gtItem.sum() == 0:
                    continue
                gtItem = gtItem.reshape(-1, 5)

                wywh = gtItem[0, :4]
                x1, y1 = wywh[0:2]
                x2, y2 = wywh[2:4]

                gtItem[:, 0] = (x1 + x2) / 2
                gtItem[:, 1] = (y1 + y2) / 2
                gtItem[:, 2] = (x2 - x1)
                gtItem[:, 3] = (y2 - y1)

                # # get offset
                cx, cy = grids * gtItem[0, :2]
                w, h, class_Num = gtItem[0, 2:]
                gx, gy = cx.long(), cy.long()
                offx = cx - gx
                offy = cy - gy
                offw, offh = w, h

                # convert cxcywh to x1y1x2y2 format
                pred_box = predcxywh[bszie, :, gy, gx][..., 1:]  # [bi, 2, 4]
                pred_xy = torch.FloatTensor(pred_box.size())
                pred_xy[:, 0:2] = pred_box[:, :2] / float(grids) - 0.5 * pred_box[:, 2:4]
                pred_xy[:, 2:4] = pred_box[:, :2] / float(grids) + 0.5 * pred_box[:, 2:4]

                target_xy = torch.zeros(1, 4)
                target_xy[:, 0] = offx / float(grids) - 0.5 * offw
                target_xy[:, 1] = offy / float(grids) - 0.5 * offh
                target_xy[:, 2] = offx / float(grids) + 0.5 * offw
                target_xy[:, 3] = offy / float(grids) + 0.5 * offh

                # # 真实标签和预测结果的最佳匹配
                iou_score = self.Batch_Iou(pred_xy, target_xy)
                max_iou, max_index = iou_score.max(0)
                # if iou_score.sum() == 0: print("iou all is 0")

                confxywh[bszie, max_index, gy, gx, 0] = 1
                confxywh[bszie, max_index, gy, gx, 1] = offx
                confxywh[bszie, max_index, gy, gx, 2] = offy
                confxywh[bszie, max_index, gy, gx, 3] = offw
                confxywh[bszie, max_index, gy, gx, 4] = offh

                class_Num = int(class_Num)
                clcgt[bszie, gy, gx, class_Num] = 1

        return confxywh, clcgt

    def Batch_Iou(self, box1, box2):
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)

        return iou
