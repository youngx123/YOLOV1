# -*- coding: utf-8 -*-
# @Author : xyoung

import gc
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import DataLoader
import logging
import math
from model.YOLO import YOLOV1
from dataloader.dataloader2 import yoloDataset
from model.yolo_loss import yoloLoss
from model.utils import ModelEMA
# from model.Loss import Loss as yoloLoss

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('--class_num', default=20, type=int, help='dataset class number')
    parser.add_argument('--EPOCH', default=120, type=int, help='train epoch')
    parser.add_argument('--train_size', default=416, type=int, help='train epoch')

    parser.add_argument('--train_text', default="./2007train.txt", type=str, help='train list text')
    parser.add_argument('--val_text', default="./2007test.txt", type=str, help='validate list text')

    parser.add_argument('--batch_size', default=12, type=int, help='Batch size for training')

    parser.add_argument('--val_batch_size', default=2, type=int, help='Batch size for training')

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch to train')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--warm_up', default=5, type=int, help='lr warm up step')

    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda.')

    parser.add_argument('--eval_epoch', type=int, default=4, help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str, help='pt save dir')
    parser.add_argument('--show_loss_step', default=1, type=int, help='show training info')

    parser.add_argument('--pretrained_model', default="./weights/model.pt", type=str, help='load pretrained model')

    return parser.parse_args()


def save_checkpoint(state_dict, optimizer, config, modelName):
    """
    state_dict : net state dict
    optimizer: optimizer parameters
    epoch : train epoch
    config : config files
    """
    checkpoint_path = os.path.join(config.save_folder, f"{modelName}.pt")
    model_dict = {
        "model_weight": state_dict,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(model_dict, checkpoint_path)
    return checkpoint_path


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(config, net, loss_fn, optimizer, trainloader, valloader):
    EPOCH = config.EPOCH
    device = torch.device("cuda")
    baseLr = 0.001
    net.to(device)
    net.train()
    best_val = np.inf

    nb = len(trainloader)  # number of batches
    n_burn = max(config.warm_up * nb, 500)
    lf = lambda x: (((1 + math.cos(x * math.pi / EPOCH)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
    ema = ModelEMA(net)
    for epoch in range(EPOCH):
        progressBar = tqdm(trainloader)
        for step, batch in enumerate(progressBar):
            img, label = batch[0].float().to(device), batch[1].float().to(device)

            ni = step + nb * epoch  # number integrated batches (since train start)
            # # warm-up
            # if ni <= n_burn * 2:
            if ni <= n_burn:
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, baseLr * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, 0.937])
            if ni == n_burn:
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = baseLr

            pred = net(img)
            sum_loss = loss_fn(pred, label)

            optimizer.zero_grad()
            sum_loss.backward()
            optimizer.step()
            progressBar.set_description("Epoch: %d , loss : %.3f , lr : %.8f" %
                                        (epoch, sum_loss.item(),
                                         [param_group['lr'] for param_group in optimizer.param_groups][0]))
        ema.update(net)
        lr_scheduler.step()
        # # start eval and save net state dict
        if epoch % config.eval_epoch == 0:
            net.eval()
            checkpointModel = save_checkpoint(net.state_dict(), optimizer, config, "model")
            logging.info("Model checkpoint saved to :%s" % checkpointModel)
            val_loss = eval_model(net, loss_fn, valloader)
            if best_val > val_loss:
                logging.info("validate loss improve from  %.4f to  %.4f " % (best_val, val_loss))
                best_val = val_loss

                checkpointBest = save_checkpoint(net.state_dict(), optimizer, config, "best")
                logging.info("Best checkpoint saved to : %s" % checkpointBest)
            else:
                logging.warning("validate loss dont improve best :  %.4f , val loss : %.4f " % (best_val, val_loss))
        net.train()

    checkpointModel = save_checkpoint(net.state_dict(), optimizer, config, "last")
    logging.info("Model checkpoint saved to :%s" % checkpointModel)


def eval_model(model, Loss_Func, valloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for step, samples in enumerate(valloader):
            img, label = samples[0].float().to("cuda"), samples[1].float().to("cuda")
            outputs = model(img)
            _loss_item = Loss_Func(outputs, label)
            loss += _loss_item
        loss = loss / len(valloader)
    return loss.item()


def main(config):
    train_size = [config.train_size, config.train_size]

    # set train and val data text
    train_Text = config.train_text
    val_Text = config.val_text

    # set train and val batch size
    batch_size = config.batch_size
    val_batch_size = config.val_batch_size

    # class number
    classNum = config.class_num
    net = YOLOV1(v1head=True, class_num=classNum)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if config.pretrained_model:
        net_state_dict = torch.load(config.pretrained_model)
        trainedDict = net_state_dict["model_weight"]
        netDict = net.state_dict()
        for k in trainedDict.keys():
            if k in netDict.keys() and trainedDict[k].shape == netDict[k].shape:
                netDict[k] = trainedDict[k]
        net.load_state_dict(netDict)
        logging.info("load pretrained model")

        optDict = net_state_dict["optimizer"]
        # for k in optDict.keys()
        optimizer.load_state_dict(optDict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        del net_state_dict, trainedDict, netDict
        gc.collect()
        torch.cuda.empty_cache()

    traindataset = yoloDataset(train_Text, imageSie=train_size[0], classNum=classNum, train=True)
    val_dataset = yoloDataset(val_Text, imageSie=train_size[0], classNum=classNum, train=True)
    trainLoader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    val_Loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    loss_fn = yoloLoss()

    train_model(config, net, loss_fn, optimizer, trainLoader, val_Loader)


if __name__ == '__main__':
    config = parse_args()
    main(config)
