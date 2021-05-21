# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
import torch.nn.functional as F
from torchvision import transforms


def forward(input):
    pyramids = [1, 2, 3, 6]
    feat = input
    height, width = input.shape[2:]

    for bin_size in pyramids:
        x = F.adaptive_avg_pool2d(input, output_size=bin_size)

        inputsz = np.array(input.shape[2:])
        outputsz = np.array([bin_size, bin_size])
        stridesz = np.floor(inputsz / outputsz).astype(np.int32)
        kernelsz = inputsz - (outputsz - 1) * stridesz
        avg = torch.nn.AvgPool2d( kernel_size=list(kernelsz), stride=list(stridesz))
        y = avg(input)

        tmp_x = np.around(x.numpy(), 3)
        tmp_y = np.around(y.numpy(), 3)
        #print(x-y)
        print(bin_size, 'adp vs avg: ', (tmp_x == tmp_y).all())
        print("========avg para kernelsz, stridesz======:",list(kernelsz), list(stridesz))
        print(x.dtype,x.shape)
        print(y.dtype, y.shape)

        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
        feat = feat + x
    return feat

if __name__ == '__main__':

    # input = torch.randn(1, 2048, 32, 64)
    np_input = np.load('./input.npy')
    print(np_input.shape,np_input.dtype)
    tensor_input = torch.from_numpy(np_input)
    print(tensor_input.shape,tensor_input.dtype)
    feat = forward(tensor_input)
    print(feat.shape,feat.dtype)
