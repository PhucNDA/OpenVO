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

def project_depth_map_batch(depth_map1, depth_map2, optical_flow, intrinsics):
    fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]
    BS, H, W = depth_map1.shape
    
    u = torch.arange(W, device=depth_map1.device).unsqueeze(0).expand(H, W)
    v = torch.arange(H, device=depth_map1.device).unsqueeze(1).expand(H, W)
    u = u.unsqueeze(0).expand(BS, H, W)
    v = v.unsqueeze(0).expand(BS, H, W)

    d = depth_map1
    X = (u - cx.view(-1, 1, 1)) * d / fx.view(-1, 1, 1)
    Y = (v - cy.view(-1, 1, 1)) * d / fy.view(-1, 1, 1)
    Z = d
    depth_1 = torch.stack((X, Y, Z), dim=1)  
    
    x = (u + optical_flow[:, 0]).clone().detach().to(torch.int).to(depth_map1.device)
    y = (v + optical_flow[:, 1]).clone().detach().to(torch.int).to(depth_map1.device)
    
    grid = torch.stack((x / (W - 1) * 2 - 1, y / (H - 1) * 2 - 1), dim=-1)
    d2 = F.grid_sample(depth_map2.unsqueeze(1), grid, mode='bilinear', align_corners=True).squeeze(1)
    d2[d2 == 0] = d[d2 == 0]

    X2 = (u - cx.view(-1, 1, 1)) * d2 / fx.view(-1, 1, 1)
    Y2 = (v - cy.view(-1, 1, 1)) * d2 / fy.view(-1, 1, 1)
    Z2 = d2
    depth_2 = torch.stack((X2, Y2, Z2), dim=1) 
    
    scene_flow = depth_2 - depth_1 

    return scene_flow, 1

def differentiable_depth_map_batch(depth1, depth2, flow, intrinsics, align_corners=True):
    """
    depth1, depth2: (B, H, W) in meters
    flow: (B, 2, H, W) pixel displacement from t0->t1 (x,y)
    intrinsics: (B, 4) as (fx, fy, cx, cy)
    Returns:
        scene_flow: (B, 3, H, W) in camera-1 coords  (X2 - X1)
        valid: (B, 1, H, W) boolean mask
    """
    B, H, W = depth1.shape
    fx, fy, cx, cy = [intrinsics[:, i].view(B, 1, 1) for i in range(4)]

    # base pixel grid (u,v)
    u = torch.linspace(0, W - 1, W, device=depth1.device).view(1, 1, W).expand(B, H, W)
    v = torch.linspace(0, H - 1, H, device=depth1.device).view(1, H, 1).expand(B, H, W)

    # back-project at t0
    Z1 = depth1
    X1 = (u - cx) * Z1 / fx
    Y1 = (v - cy) * Z1 / fy

    # warp pixels with flow (subpixel, differentiable)
    u1 = u + flow[:, 0]
    v1 = v + flow[:, 1]

    # normalized grid for grid_sample
    if align_corners:
        norm_u1 = (u1 / (W - 1)) * 2 - 1
        norm_v1 = (v1 / (H - 1)) * 2 - 1
    else:
        norm_u1 = ((u1 + 0.5) / W) * 2 - 1
        norm_v1 = ((v1 + 0.5) / H) * 2 - 1

    grid = torch.stack([norm_u1, norm_v1], dim=-1)  # (B, H, W, 2)

    # sample depth2 at warped locations
    Z2 = F.grid_sample(depth2.unsqueeze(1), grid, mode='bilinear',
                       padding_mode='zeros', align_corners=align_corners).squeeze(1)

    # valid pixels: inside image & positive finite depth
    inside = (u1 >= 0) & (u1 <= W - 1) & (v1 >= 0) & (v1 <= H - 1)
    valid = inside & torch.isfinite(Z1) & torch.isfinite(Z2) & (Z1 > 0) & (Z2 > 0)
    valid = valid.unsqueeze(1)

    # back-project at t1 using WARPED pixels (u1, v1)
    X2 = (u1 - cx) * Z2 / fx
    Y2 = (v1 - cy) * Z2 / fy

    scene_flow = torch.stack([X2 - X1, Y2 - Y1, Z2 - Z1], dim=1)  # (B, 3, H, W)
    scene_flow = scene_flow * valid  # mask out invalid
    return scene_flow, valid




class ZVOModel(nn.Module):
    def __init__(self, args):
        super(ZVOModel, self).__init__()
        
        self.args = args
        '''
        Encoder
        '''
        self.encoder = FlowNetS(self.args)

        '''
        Feature shape for LSTM
        '''
        
        optical_flow_feature_size = int((96/self.args.feat_patch_size[0])*(160/self.args.feat_patch_size[1])*self.args.attn_embed_dim)
        depth_intrs_feature_size = int((self.args.img_h/self.args.img_patch_size[0])*(self.args.img_w/self.args.img_patch_size[1])*self.args.attn_embed_dim)
        '''
        Decoder
        '''
        print('The size of optical_flow_feature is', optical_flow_feature_size, 'The size of depth_text_feature is', depth_intrs_feature_size)
        self.decoder = LSTM(self.args, optical_flow_feature_size + depth_intrs_feature_size)

        flow_2d_params = dict(img_size=(96, 160), patch_size=self.args.feat_patch_size, embed_dim=self.args.attn_embed_dim, depth=4, num_heads=4, num_classes=0, in_chans=102, class_token=False, global_pool='')
        flow_3d_params = dict(img_size=(self.args.img_h, self.args.img_w), patch_size=self.args.img_patch_size, embed_dim=self.args.attn_embed_dim, depth=4, num_heads=4, num_classes=0, in_chans=3, class_token=False, global_pool='', )


        # From pretrained
        self.flow_2d_transformer = _create_vision_transformer("vit_base_patch16_224_in21k", pretrained=False, **flow_2d_params)
        self.flow_2d_transformer = create_feature_extractor(self.flow_2d_transformer, return_nodes={"norm": "feature"}) 

        self.flow_3d_transformer = _create_vision_transformer("vit_base_patch16_224_in21k", pretrained=False, **flow_3d_params)
        self.flow_3d_transformer = create_feature_extractor(self.flow_3d_transformer, return_nodes={"norm": "feature"})

        self.dropout1 = nn.Dropout(0.1)
        self.patch_embedding1 = PatchEmbedding([self.args.img_h, self.args.img_w], patch_size=128, num_hiddens=768)
        self.pos_embedding1 = nn.Parameter(torch.randn(1, self.patch_embedding1.num_patches, 768))
        # From scratch
        self.depth3d_blocks = nn.ModuleList([Block(768, self.args.num_heads, 0.2, 0.2) for _ in range(self.args.num_blocks)])


    def forward(self, x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs):
        
        batch_size = x.size(0)
        x, flow_2d = self.encoder(x) # x = (16, 102, 96, 160)
        
        ## Diff Scene Flow
        if self.args.diffflow == True:
            flow_3d, _ = differentiable_depth_map_batch(depth_map0[:,0,:,:], depth_map1[:,0,:,:], flow_2d, intrs, align_corners= True) # (16, 3, 384, 640)
        else:
            flow_3d, _ = project_depth_map_batch(depth_map0[:,0,:,:], depth_map1[:,0,:,:], flow_2d, intrs)
        ## self-attn
        flow_2d_feature = self.flow_2d_transformer(x)['feature'] # (16, 15, 768)
        flow_3d_feature = self.flow_3d_transformer(flow_3d)['feature'] # (16, 15, 768)
        ## Flow Aggregation
        flow_feature = flow_2d_feature + flow_3d_feature

        f1 = self.patch_embedding1(depth_3d)
        f1 = self.dropout1(f1 + self.pos_embedding1)

        depth_3d_feature = f1
        for block in self.depth3d_blocks:
            ## self-attn
            depth_3d_feature = block(depth_3d_feature, depth_3d_feature)
        
        return self.decoder(torch.cat((flow_feature.reshape(batch_size, -1), depth_3d_feature.reshape(batch_size, -1)), dim=1))
    
    def get_loss(self, x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs):

        pose, A = self.forward(x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs)

        losses, pred_orth = fisher_NLL(A, y[:,6:], overreg=1.025)
        loss_A = losses.mean()

        loss_p = torch.nn.functional.mse_loss(pose[:,:], y[:,:6])

        return loss_p, loss_A

    def step(self, x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs):

        loss_p, loss_A = self.get_loss(x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs)
        loss = loss_p + loss_A
        loss.backward()

        return loss_p
