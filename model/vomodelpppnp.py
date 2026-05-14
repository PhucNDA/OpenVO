from .flownet import FlowNetS
from .lstm import LSTM
from .cross_attn import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
# from timm.models.vision_transformer_cross import _create_vision_transformer
from timm.models.vision_transformer import _create_vision_transformer

import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from fisher.fisher_utils import vmf_loss as fisher_NLL
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FlowPnP(nn.Module):
    def __init__(self, args):
        super(FlowPnP, self).__init__()
        
        self.args = args
        '''
        Encoder
        '''
        self.encoder = FlowNetS(self.args)

        '''
        Feature shape for LSTM
        '''
        
        optical_flow_feature_size = int((96/self.args.feat_patch_size[0])*(160/self.args.feat_patch_size[1])*self.args.attn_embed_dim)

        print('The size of optical_flow_feature is', optical_flow_feature_size,)


    def forward(self, x):
        
        batch_size = x.size(0)
        x, flow_2d = self.encoder(x) # x = (16, 102, 96, 160)
        
        return x, flow_2d
