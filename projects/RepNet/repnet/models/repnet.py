#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file repnet.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-05-12 14:22


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as M
from torchvision import transforms as T


class ResNet50Base5D(nn.Module):
    def __init__(self, pretrained=False, m=2):
        super().__init__()
        base_model = M.resnet50(pretrained=pretrained)

        if m == 1:
            # method-1:
            base_model.fc = nn.Identity()
            base_model.avgpool = nn.Identity()
            base_model.layer4 = nn.Identity()
            base_model.layer3[3] = nn.Identity()
            base_model.layer3[4] = nn.Identity()
            base_model.layer3[5] = nn.Identity()
            self.base_model = base_model
        else:
            # method-2:
            self.base_model = nn.Sequential(
                *list(base_model.children())[:-4],
                *list(base_model.children())[-4][:3])

    def forward(self, x):
        N, S, C, H, W = x.shape
        x = x.view(-1, C, H, W) # 5D -> 4D
        x = self.base_model(x)
        x = x.view(N, S, x.shape[1], x.shape[2], x.shape[3]) # 4D -> 5D
        return x


class TemporalContext(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()
        self.conv3D = nn.Sequential(
            nn.Conv3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                padding = 3,
                dilation = 3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        # (N, S, C, H, W) -> (N, C, S, H, W)
        x = x.transpose(1, 2)
        x = self.conv3D(x)
        return x


class GlobalMaxPool(nn.Module):
    def __init__(self, m=2):
        super().__init__()
        self.m = m

        # method:2
        self.pool = nn.MaxPool3d(kernel_size = (1, 7, 7))

    def forward(self, x):
        # Inputs: (N, C, S, 7, 7)
        # method:1
        if self.m == 1:
            x, _ = torch.max(x, dim=4)
            x, _ = torch.max(x, dim=3)
        else:
            # method:2
            x = self.pool(x).squeeze(-1).squeeze(-1)

        x = x.transpose(1, 2)
        return x # (N, S, C)

class TemproalSelfMatrix(nn.Module):
    def __init__(self, temperature=13.544, m=2):
        super().__init__()
        self.temperature = temperature
        self.m = m

    def calc_sims(self, x):
        # (N, S, E)  --> (N, 1, S, S)
        S = x.shape[1]

        I = torch.ones(S).to(x.device)
        xr = torch.einsum('nse,h->nhse', (x, I))
        xc = torch.einsum('nse,h->nshe', (x, I))
        diff = xr - xc
        return torch.einsum('nsge,nsge->nsg', (diff, diff))

    def pairwise_l2_distance(self, x):
        # (S, E)
        a, b = x, x
        norm_a = torch.sum(torch.square(a), dim=1)
        norm_a = torch.reshape(norm_a, [-1, 1])
        norm_b = torch.sum(torch.square(b), dim=1)
        norm_b = torch.reshape(norm_b, [1, -1])
        b = torch.transpose(b, 0, 1)  # a: 64x512  b: 512x64
        zero_tensor = torch.zeros(64, 64)
        dist = torch.maximum(norm_a - 2.0 * torch.matmul(a, b) + norm_b, torch.tensor(0.0).to(x.device))
        return dist

    def forward(self, x):
        # x: (N, S, E)
        # method: 1
        if self.m == 1:
            # x = torch.transpose(x, 1, 2)
            sims_list = []
            for i in range(x.shape[0]):
                sims_list.append(self.pairwise_l2_distance(x[i]))
            sims = torch.stack(sims_list)
        else:
            # method: 2
            sims = self.calc_sims(x)

        sims = sims.unsqueeze(1)
        sims = F.softmax(-sims/self.temperature, dim=-1)
        return F.relu(sims) # (N, 1, S, S)


class FeaturesProjection(nn.Module):
    def __init__(self, num_frames=64, out_features=512, dropout=0.2):
        super().__init__()
        self.num_frames = num_frames
        self.out_features = out_features
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.projection = nn.Sequential(
            nn.Linear(num_frames*32, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features))

    def forward(self, x):
        x = self.features(x) # N, 32, S, S
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), self.num_frames, -1) # N, S, 32*S
        x = self.projection(x) # N, S, 512
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # (S, N, 512)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    def __init__(self, num_frames=64, d_model=512, 
                 n_head=4, dim_ff=512, dropout=0.2,
                 num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, num_frames)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
                
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        return x


class PeriodClassifier(nn.Module):
    def __init__(self, num_frames=64, in_features=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.LayerNorm(512),
            nn.Linear(in_features=512, out_features=num_frames//2),
            nn.Linear(in_features=num_frames//2, out_features=1))

    def forward(self, x):
        x = self.classifier(x)
        return x


class RepNet(nn.Module):
    def __init__(self, num_frames=64, num_dmodel=512):
        super().__init__()
        ## Encoder
        self.resnet50 = ResNet50Base5D(pretrained=True)
        self.tcxt = TemporalContext()
        self.maxpool = GlobalMaxPool(m=2)
        ## TSM
        self.tsm = TemproalSelfMatrix(temperature=13.544, m=2)
        ## Period Predictor
        self.projection = FeaturesProjection(num_frames=num_frames, out_features=num_dmodel, dropout=0.0)
        ### period length prediction
        self.trans1 = TransformerModel(num_frames, d_model=num_dmodel, n_head=4,
                                   dropout=0.1, dim_ff=num_dmodel)
        self.pc1 = PeriodClassifier(num_frames, num_dmodel)
        ### periodicity prediction
        self.trans2 = TransformerModel(num_frames, d_model=num_dmodel, n_head=4,
                                   dropout=0.1, dim_ff=num_dmodel)
        self.pc2 = PeriodClassifier(num_frames, num_dmodel)
        
    def forward(self, x, retsim=False):
        x = self.resnet50(x)
        x = self.tcxt(x)
        x = self.maxpool(x)
        x = self.tsm(x)
        if retsim:
            z = x
        x = self.projection(x)
        
        y1 = self.pc1(self.trans1(x))
        y2 = self.pc2(self.trans2(x))
        if retsim:
            return y1, y2, z
        else:
            return y1, y2
