# -*- coding: utf-8 -*-
# @Author : Young

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import math
from copy import deepcopy


# # copy from yolov5
# new_average = (1.0 - mu) * x + mu * self.shadow[name]
class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updateStep = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updateStep += 1
            d = self.decay(self.updateStep)  # # 计算当前的衰减率

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    msd[k] = v

            model.load_state_dict(msd)


"""
class EMA():
    def __init__(self,  decay):
        self.decay = decay
        # 用于保存上一次迭代计算出来的结果
        self.shadow = {}

    def register(self,name,val):
        self.shadow[name] = val.clone()

    def __call__(self,name,x):
        assert name in self.shadow
        new_average=self.decay*x+(1.0-self.decay)*self.shadow[name]
        self.shadow[name]=new_average.clone()
        return new_average

# 初始化
ema = EMA(0.999)
for name, param in model.named_parameters():
   if param.requires_grad:
       ema.register(name, param.data)

# 在batch中
for batch in batches:
   optimizer.step()
   for name,param in model.named_parameters():
       if param.requires_grad:
           param.data=ema(name,param.data)
"""


def decay(b, i):
    for k, v in b.items():
        b[k] = v * i


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=False)
        self.b1 = torch.nn.BatchNorm2d(6)
        self.c2 = torch.nn.Conv2d(6, 6, 3, 2, 1, bias=False)
        self.b2 = torch.nn.BatchNorm2d(6)

    def forward(self, x):
        x = self.c(x)
        x = self.b1(x)
        x = self.b2(self.c2(x))
        return x


if __name__ == '__main__':
    a = {"1": 1, "2": 2, "3": 3,
         "4": 4, "5": 5, "6": 6}

    decay(a, 0.1)
    decay(a, 2)
    for k, v in a.items():
        print(k, v)

    net = Net()
    net.train()
    ema = ModelEMA(net)

    ema.update(net)

    print(net)
