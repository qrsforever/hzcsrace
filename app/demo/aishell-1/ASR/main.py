#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-03-30 22:45


import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Test(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, shape):
        test = torch.zeros(*shape)
        return None

if __name__ == '__main__':

    model = nn.DataParallel(Test()).to(device)
    model.train()

    for i in range(1000):
        for j in range(1000):
            shape = (8, 850 + i, 180 + j)
            model(shape)
