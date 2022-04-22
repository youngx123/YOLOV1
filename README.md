
学习yolov1标签分配的思想以及按照自己理解重写损失函数

```python
w, h 是相对于训练图像的大小，而cx,cy是相对于特征图的大小，因此在计算iou 时，应将cx,cy也转
换为相对于图像大小，最终，标签和预测结果的iou计算都是使用相对于输入图像大小。

cxy = box / width ==> 坐标归一化
cxcy = cxy *gridSize ==>在特征图上的坐标大小

cxy = cxcy / gridSize ==> 特征图上坐标和训练图像坐标关系

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
```

>https://github.com/abeardear/pytorch-YOLO-v1
>
>
