import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model.YOLO import YOLOV1
import numpy as np
import cv2
import torchvision
import dataloader.dataloader2 as dataloader2
from dataloader.dataloader2 import yoloDataset
import time
import imageio

VOC_CLASSES = dataloader2.VOC_CLASSES

parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('--dataset', default='2007test.txt',
                    help='test file names txt')
parser.add_argument('--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default=r'D:\MyNAS\SynologyDrive\yolov1\weights\model.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')

args = parser.parse_args()


def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    if dataset == 'voc':
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 2)
                cv2.rectangle(img, (int(xmin), int(abs(ymin) - 20)), (int(xmax), int(ymin)),
                              class_colors[int(cls_indx)], -1)
                mess = '%s' % (class_names[int(cls_indx)])
                cv2.putText(img, mess, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 2)
                cv2.rectangle(img, (int(xmin), int(abs(ymin) - 20)), (int(xmax), int(ymin)),
                              class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def Bbox_IOU(b1, b2):
    b1 = b1.reshape(-1, 4)
    xmin = torch.max(b1[:, 0], b2[:, 0])
    xmax = torch.min(b1[:, 2], b2[:, 2])
    ymin = torch.max(b1[:, 1], b2[:, 1])
    ymax = torch.min(b1[:, 3], b2[:, 3])

    wlenth = xmax - xmin
    hlenth = ymax - ymin

    union = wlenth * hlenth
    area1 = (b1[:, 3] - b1[:, 1]) * (b1[:, 2] - b1[:, 0])
    area2 = (b2[:, 3] - b2[:, 1]) * (b2[:, 2] - b2[:, 0])
    iou = union / (area1 + area2 - union +1e-6)
    return iou


def NMS(det_class, nms_thres):
    nms_result = []
    while (det_class.size(0)):
        nms_result.append(det_class[0].unsqueeze(0))
        if len(det_class) == 1:
            break
        det_class = det_class[1:]
        ious = Bbox_IOU(nms_result[-1], det_class)

        # Remove detections with IoU >= NMS threshold
        det_class = det_class[ious < nms_thres]
    return nms_result


def test(net, device, testset, thresh,
         class_colors=None, class_names=None, class_indexs=None):
    num_images = len(testset)

    for step, batch in enumerate(testset):
        img, path = batch[0].float().to(device), batch[1]
        orig_img = imageio.imread(path[0])
        origH, origW = orig_img.shape[:2]
        pred = net(img)

        pred = pred[0]
        pred_conf = torch.sigmoid(pred[..., 0])
        pred_xywh = pred[..., 1:5]
        pred_xywh[..., :2] = torch.sigmoid(pred_xywh[..., :2]).clip(0, 1)

        grid_size = pred_xywh.shape[0]
        x = pred_xywh[..., 0]
        FloatTensor = torch.cuda.FloatTensor if pred_xywh.is_cuda else torch.FloatTensor
        xGrid = torch.linspace(0, grid_size - 1, grid_size).repeat(grid_size, 1).repeat(
                                int(1 * 1), 1, 1).view(x.shape).type(FloatTensor)
        yGrid = torch.linspace(0, grid_size - 1, grid_size).repeat(grid_size, 1).t().repeat(
                                int(1 * 1), 1, 1).view(x.shape).type(FloatTensor)

        pred_xywh[..., 0] = pred_xywh[..., 0] + xGrid
        pred_xywh[..., 1] = pred_xywh[..., 1] + yGrid
        pred_xywh[..., 2] = torch.exp(pred_xywh[..., 2])
        pred_xywh[..., 3] = torch.exp(pred_xywh[..., 3])

        pred_clc = torch.sigmoid(pred[..., 5:])
        pred_clc = pred_clc.view(-1,20)
        max_prob, cls_index = torch.max(pred_clc, 1, keepdim=True)
        pred_confa = pred_conf.view(-1,1)
        scores = pred_confa[:,0] * max_prob[:,0]
        objMask = scores > 0.5
        pred_xywh = pred_xywh.view(-1,4)
        score = scores[objMask]
        scoreSort, index = torch.sort(score, descending=True)

        pred_xywh = pred_xywh[objMask]
        boxs = torch.zeros_like(pred_xywh)
        boxs[..., 0] = pred_xywh[..., 0] - pred_xywh[..., 2]/2
        boxs[..., 1] = pred_xywh[..., 1] - pred_xywh[..., 3]/2
        boxs[..., 2] = pred_xywh[..., 0] + pred_xywh[..., 2]/2
        boxs[..., 3] = pred_xywh[..., 1] + pred_xywh[..., 3]/2
        pred_xywh = boxs*32
        pred_xywh = pred_xywh[index]

        result = NMS(pred_xywh, 0.4)
        # write result images
        if len(result)>0:
            # result = result[0].cpu().numpy().reshape(-1, 7)
            for idx, detections in enumerate(result):
                detections = detections.detach().numpy()
                x1, y1, x2, y2 = detections[0]
                # box_h = np.ceil(((y2 - y1) / 416) * origin_size[1])
                # box_w = np.ceil(((x2 - x1) / self.IMAGE_SIZE) * origin_size[0])
                y1 = np.ceil((y1 / 416) * origH)
                x1 = np.ceil((x1 / 416) * origW)

                y2 = np.ceil((y2 / 416) * origH)
                x2 = np.ceil((x2 / 416) * origW)

                # mess = "%s:%.2f" % (CLASS_NAME[int(cls_pred)], round(conf, 2))
                cv2.rectangle(orig_img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 0, 0), 2)
                # cv2.putText(images_data22, mess, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             (0, 255, 0), 1)

            imageio.imsave('{}.png'.format(step), (orig_img).astype(np.uint8))


if __name__ == '__main__':
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = [args.input_size, args.input_size]

    # dataset
    print('test on voc ...')
    class_names = VOC_CLASSES
    class_indexs = None
    num_classes = 20
    dataset = yoloDataset(args.dataset, imageSie=input_size[0], train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    class_colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
                    range(num_classes)]

    # build model
    net = YOLOV1(v1head=True, class_num=20)
    if args.trained_model:
        net_state_dict = torch.load(args.trained_model)
        net.load_state_dict(net_state_dict["model_weight"])
        print("load pretrained model")

    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test(net=net, device=device, testset=dataloader, thresh=args.visual_threshold,
         class_colors=class_colors, class_names=class_names, class_indexs=class_indexs)
