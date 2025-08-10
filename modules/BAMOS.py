# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *

from modules.BaseBlocks import MetaKernel, ResContextBlock, ResBlock, UpBlock
from typing import Tuple
from copy import deepcopy

class FeatureAttentionFusion(nn.Module):
    def __init__(self, channels, reduction=2):
        super(FeatureAttentionFusion, self).__init__()
        
        reduced_channels = channels // reduction

        
        self.conv1 = nn.Conv2d(channels + reduced_channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(reduced_channels, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_i, y_i):
        
        z_i = torch.cat([x_i, y_i], dim=1)  
        
        f_i = F.relu(self.conv1(z_i))  
        
        a_i = self.sigmoid(self.conv2(f_i))  
       
        x_i = self.conv3(x_i)
        weighted_x = x_i * (1 - a_i)
        weighted_y = y_i * a_i

        
        fused_features = self.fusion_conv(weighted_x + weighted_y)

        return fused_features

class CustomAttentionModule(nn.Module):
    def __init__(self, in_channels_1, in_channels_2):
        super(CustomAttentionModule, self).__init__()
        
        # First 3x3 Conv, ReLU, BatchNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_1 + in_channels_2, in_channels_2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels_2)
        )
        
        # Second 3x3 Conv, BatchNorm, ReLU, 3x3 Conv, BatchNorm, Sigmoid
        self.attention_branch = nn.Sequential(
            nn.Conv2d(in_channels_2, in_channels_2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_2, in_channels_2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels_2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Concatenate along the channel dimension
        x_cat = torch.cat((x1, x2), dim=1)  # Shape: (B, C1 + C2, H, W)

        # Apply first branch
        out1 = self.conv1(x_cat) + x2 # Shape: (B, C, H, W)

        # Apply attention branch
        attention = self.attention_branch(out1)  # Shape: (B, C, H, W)

        # Element-wise multiplication (attention weighting)
        out2 = out1 * attention  # Shape: (B, C, H, W)

        # Residual connection
        out = out1 + out2  # Shape: (B, C, H, W)

        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, flow):
        x = torch.cat([torch.mean(img, dim=1, keepdim=True),
                       torch.max(img, dim=1, keepdim=True)[0],
                       torch.mean(flow, dim=1, keepdim=True),
                       torch.max(flow, dim=1, keepdim=True)[0]], dim=1)
        x = self.sigmoid(self.conv1(x))
        return img.mul(x)+flow.mul(1-x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio): 
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes*2, in_planes // ratio, 1)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)
        self.in_planes=in_planes
    def forward(self, img, flow):
        f = self.fc2(self.relu1(self.fc1(self.avg_pool(torch.cat([img, flow], 1)))))#B 2C 1 1
        f=F.sigmoid(f)
        return img.mul(f), flow.mul(1-f)


class STAFM(nn.Module):
    def __init__(self, inplanes, ratio=2): # ratio:4 2 1
        super(STAFM, self).__init__()
        self.channel = ChannelAttention(inplanes, ratio=ratio)
        self.spatial = SpatialAttention()
    def forward(self, spatial, temporal):
        spatial, temporal = self.channel(spatial, temporal)
        feature = self.spatial(spatial, temporal)
        return feature



class BAMOS(nn.Module):
    def __init__(self, nclasses, movable_nclasses, params, num_batch=None, point_refine=None):
        super(BAMOS, self).__init__()
        self.nclasses = nclasses
        self.use_attention = "MGA"
        self.train_mode = False
        self.point_refine = point_refine

        self.range_channel = 5
        print("Channel of range image input = ", self.range_channel)
        print("Number of residual images input = ", params['train']['n_input_scans'])
        
        self.downCntx = ResContextBlock(self.range_channel, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)
        print("params['train']['batch_size']", params['train']['batch_size'])
        self.metaConv = MetaKernel(num_batch=int(params['train']['batch_size']) if num_batch is None else num_batch,
                                   feat_height=params['dataset']['sensor']['img_prop']['height'],
                                   feat_width=params['dataset']['sensor']['img_prop']['width'],
                                   coord_channels=self.range_channel)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4))
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False, kernel_size=(2, 4))

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.range_upBlock1 = UpBlock(256, 128, 0.2)
        self.range_upBlock2 = UpBlock(128, 64, 0.2)
        self.range_upBlock3 = UpBlock(64, 32, 0.2)

        # Context Block for residual image 
        self.RI_downCntx = ResContextBlock(params['train']['n_input_scans'], 32)

        self.RI_resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4))
        self.RI_resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock5 = ResBlock(2 * 4 * 32, 4 * 4 * 32, 0.2, pooling=False, kernel_size=(2, 4))
        
        self.logits3 = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.movable_logits = nn.Conv2d(32, movable_nclasses, kernel_size=(1, 1))

        #******************************************************************#
        self.edge_conv1 = self.generate_edge_conv(32, 32)
        self.edge_conv2 = self.generate_edge_conv(64, 64)
        self.edge_conv3 = self.generate_edge_conv(128, 128)
        self.edge_conv4 = self.generate_edge_conv(256, 256)
        self.edg12 = FeatureAttentionFusion(32, reduction=2)
        self.edg23 = FeatureAttentionFusion(16, reduction=2)
        self.edg34 = FeatureAttentionFusion(8, reduction=2)

        self.feafusion = CustomAttentionModule(80, 32)

        self.edg12_ = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.edg34_ = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.dup1 = nn.Sequential(
            nn.Conv2d(8, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.dup2 = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.edge_out = nn.Sequential(nn.Conv2d(16, 1, 1),
                                       nn.Sigmoid())

        self.att_e1 = STAFM(32, 2)
        self.att_e2 = STAFM(64, 2)
        self.att_e3 = STAFM(128, 2)
        self.att_e4 = STAFM(256, 2)
        
    def generate_edge_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )
    
    @torch.jit.script
    def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
        pH = factor_hw[0]
        pW = factor_hw[1]
        y = x
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC//(pH*pW), iH*pH, iW*pW
        y = y.reshape(B, oC, pH, pW, iH, iW)
        y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
        y = y.reshape(B, oC, oH, oW)
        return y


    def forward(self, x):
        """
            x: shape [bs, c, h, w],  c = range image channel + num of residual images
            *_downCntx:[bs, .., h, w]
            RI_down0c: [bs, c', h/2, w/2]       RI_down0b:  [bs, c', h, w] 
            RI_down1c: [bs, c'', h/4, w/4]      RI_down1b:  [bs, c'', h/2, w/2] 
            RI_down2c: [bs, c'', h/8, w/8]      RI_down2b:  [bs, c'', h/4, w/4] 
            RI_down3c: [bs, c'', h/16, w/16]    RI_down3b:  [bs, c'', h/8, w/8] 
            up4e: [bs, .., h/8, w/8] 
            up3e: [bs, .., h/4, w/4]
            up2e: [bs, .., h/2, w/2]
            up1e: [bs, .., h, w]
            logits: [bs, num_class, h, w]
        """

        # print("x shape is {}".format(x.shape))
        # split the input data to range image (5 channel) and residual images
        current_range_image = x[:, :self.range_channel, : ,:]
        residual_images = x[:, self.range_channel:, : ,:]

        ###### the Encoder for residual image ######
        RI_downCntx = self.RI_downCntx(residual_images)

        ###### the Encoder for range image ######       # range (3, 5, 64, 2048)
        downCntx = self.downCntx(current_range_image)   # (3, 32, 64, 2048)
        # Use MetaKernel to capture more spatial information
        downCntx = self.metaConv(data=downCntx,
                                 coord_data=current_range_image,
                                 data_channels=downCntx.size()[1],
                                 coord_channels=current_range_image.size()[1],
                                 kernel_size=3)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx) # b x 32 x 64 x 2048

        Range_down0c, Range_down0b = self.RI_resBlock1(downCntx)        # (64, 32, 512), (64, 64, 2048)
        Range_down1c, Range_down1b = self.RI_resBlock2(Range_down0c)    # (128, 16, 256), (128, 32, 512)
        Range_down2c, Range_down2b = self.RI_resBlock3(Range_down1c)    # (256, 8, 128) , (256, 16, 256)
        Range_down3c, Range_down3b = self.RI_resBlock4(Range_down2c)    # (256, 4, 64) , (256, 8, 128)

        ###### Bridging two specific branches using MotionGuidedAttention #####
        if self.use_attention == "MGA":
            downCntx = self.att_e1(RI_downCntx, downCntx)  # 32 x 64 x 2048
        elif self.use_attention == "Add":
            downCntx += RI_downCntx
        down0c, down0b = self.resBlock1(downCntx) # (64 x 32 x 512), (64 x 64 x 2048)

        if self.use_attention == "MGA":
            down0c = self.att_e2(down0c, Range_down0c) # 64 x 32 x 512
        elif self.use_attention == "Add":
            down0c += Range_down0c
        down1c, down1b = self.resBlock2(down0c) # (128 x 16 x 128), (128 x 32 x 512)

        if self.use_attention == "MGA":
            down1c = self.att_e3(down1c, Range_down1c) # 128 x 16 x 128
        elif self.use_attention == "Add":
            down1c += Range_down1c
        down2c, down2b = self.resBlock3(down1c) # (256 x 8 x 32), (256 x 16 x 128)

        if self.use_attention == "MGA":
            down2c = self.att_e4(down2c, Range_down2c)
        elif self.use_attention == "Add":
            down2c += Range_down2c
        down3c, down3b = self.resBlock4(down2c) # (256, 4, 8) (256, 8, 32)

        down5c = self.resBlock5(down3c)        # (256, 4, 8)

        ########edgeAttention#########
        e1 = self.edge_conv1(downCntx)

        e2 = self.edge_conv2(down0c)
        e2 = self.pixelshuffle(e2, (2, 4))
        e2 = self.dup1(e2)
        e2_out = self.edg12(e2, e1)
        
        e3 = self.edge_conv3(down1c)
        e3 = self.pixelshuffle(e3, (4, 16))
        e3 = self.dup2(e3) 
        e2_out_ = self.edg12_(e2_out)
        e3_out = self.edg23(e3, e2_out_)

        edg_out = self.edge_out(e3_out)

        e = torch.cat((e1, e2_out, e3_out), dim=1)

        ###### the Decoder, same as SalsaNext ######
        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b) # 32


        up1e = self.feafusion(e, up1e)

        logits = self.logits3(up1e)
        logits = F.softmax(logits, dim=1)
        
        if self.train_mode:
          range_up4e = self.range_upBlock1(Range_down3b, Range_down2b)
          range_up3e = self.range_upBlock2(range_up4e, Range_down1b)
          range_up2e = self.range_upBlock3(range_up3e, Range_down0b)
  
          movable_logits = self.movable_logits(range_up2e)
          movable_logits = F.softmax(movable_logits, dim=1)
        else:
          movable_logits, range_up2e = deepcopy(logits.detach()), deepcopy(up1e.detach())

        return logits, up1e, movable_logits, range_up2e, edg_out