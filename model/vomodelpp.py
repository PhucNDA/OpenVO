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
from fisher.fisher_utils import vmf_loss_omega_k
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


def time_embed_sin(dt: torch.Tensor, K: int = 8) -> torch.Tensor:
    """
    Sinusoidal embedding of a time gap dt (seconds).
    Args:
        dt: [B] or [B,1] tensor of time gaps in seconds.
        K:  number of frequency bands (log-spaced powers of 2).
    Returns:
        pe: [B, 1 + 2K]  -> [dt, sin(w_i*dt), cos(w_i*dt)]
    """
    dt = dt.view(-1, 1)                                 # [B,1]
    freqs = (2.0 ** torch.arange(K, device=dt.device))  # [K]
    w = torch.pi * freqs                                 # [K]
    sin = torch.sin(dt * w)                              # [B,K]
    cos = torch.cos(dt * w)                              # [B,K]
    pe = torch.cat([dt, sin, cos], dim=-1)               # [B,1+2K]
    return pe

def time_embed(dt: torch.Tensor) -> torch.Tensor:
    """
    Ablation embedding of a time gap dt (seconds).
    Args:
        dt: [B] or [B,1] tensor of time gaps in seconds.
    Returns:
        pe: [B, 1]  -> [dt]
    """
    dt = dt.view(-1, 1)                        # [B,1]
    pe = torch.cat([dt], dim=-1)               # [B,1+2K]
    return pe

class TimeCrossAttn2d(nn.Module):
    """
    Cross-attention from 2D feature map x (queries) to time embedding (keys/values).
    x: [B,C,H,W]
    t_emb: [B,T]  (T = tdim)
    """
    def __init__(self, channels: int, tdim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0, zero_init: bool = True):
        super().__init__()
        self.channels = channels
        self.norm_x = nn.LayerNorm(channels)
        self.norm_t = nn.LayerNorm(tdim)

        self.q_proj = nn.Linear(channels, channels, bias=True)
        self.k_proj = nn.Linear(channels, channels, bias=True)
        self.v_proj = nn.Linear(channels, channels, bias=True)
        self.out_proj = nn.Linear(channels, channels, bias=True)

        self.num_heads = num_heads
        assert channels % num_heads == 0
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # turn t_emb into 1 token in same dim as channels
        self.t_to_ctx = nn.Sequential(
            nn.Linear(tdim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        # optional feedforward (like Transformer block)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )

        if zero_init:
            nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
            # also make mlp last layer zero so it's pure identity at start
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # tokens from x
        x_tok = x.flatten(2).transpose(1, 2)          # [B, HW, C]
        x_tok = self.norm_x(x_tok)

        # 1 context token from time
        t = self.norm_t(t_emb)                        # [B, tdim]
        ctx = self.t_to_ctx(t).unsqueeze(1)           # [B, 1, C]

        # Q from x, K/V from ctx
        Q = self.q_proj(x_tok)
        K = self.k_proj(ctx)
        V = self.v_proj(ctx)

        # multi-head reshape
        Q = Q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, HW, d]
        K = K.view(B, 1,   self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, 1, d]
        V = V.view(B, 1,   self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, 1, d]

        attn = (Q * self.scale) @ K.transpose(-2, -1)                      # [B, h, HW, 1]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ V                                                     # [B, h, HW, d]
        out = out.transpose(1, 2).contiguous().view(B, H*W, C)             # [B, HW, C]
        out = self.out_proj(out)

        # residual + optional FFN
        x_tok = x_tok + out
        x_tok = x_tok + self.mlp(x_tok)

        return x_tok.transpose(1, 2).view(B, C, H, W)

class FiLM2d(nn.Module):
    """
    Feature-wise linear modulation for 2D feature maps (B,C,H,W).
    y = x * (1 + gamma(t)) + beta(t)
    """
    def __init__(self, channels: int, tdim: int, zero_init: bool = True):
        super().__init__()
        self.gamma = nn.Linear(tdim, channels)
        self.beta  = nn.Linear(tdim, channels)
        if zero_init:
            nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.gamma(t_emb).view(B, C, 1, 1)  # [B,C,1,1]
        b = self.beta(t_emb).view(B, C, 1, 1)   # [B,C,1,1]
        return x * (1 + g) + b


###### Symlog velocity
s = torch.tensor([3.6]).cuda()
y_max = torch.tensor([75.0]).cuda()  # m/s
alpha = torch.log(1 + s * abs(y_max))
def symlog(y):
    return torch.sign(y) * torch.log(1 + s * torch.abs(y)) / alpha
def inv_symlog(t):
    return torch.sign(t) * (torch.exp(torch.abs(t) * alpha) - 1) / s

class OpenVO(nn.Module):
    def __init__(self, args):
        super(OpenVO, self).__init__()
        
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
        print('The size of optical_flow_feature is', optical_flow_feature_size, 'The size of depth_feature is', depth_intrs_feature_size)
        # Ablate
        self.decoder = LSTM(self.args, optical_flow_feature_size + depth_intrs_feature_size)
        ### Flow
        # self.decoder = LSTM(self.args, optical_flow_feature_size)

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

        # Time-aware module
        K_pe = args.freq_k
        C_flowfeature = 102
        
        self.time_pe  = lambda dt: time_embed_sin(dt, K=K_pe)
        ####### Ablation
        # K_pe = 0 
        # self.time_pe  = lambda dt: time_embed(dt)
        ##############
            
        ####### Ablation # cross-attn
        # self.film_flow = TimeCrossAttn2d(
        #     channels=C_flowfeature,
        #     tdim=1+2*K_pe,
        #     num_heads=3,
        #     zero_init=True
        # )
        #############

        self.film_flow = FiLM2d(C_flowfeature, tdim=1+2*K_pe)



    def forward(self, x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs):
        
        batch_size = x.size(0)
        x, flow_2d = self.encoder(x) # x = (16, 102, 96, 160)
        
        t_emb = self.time_pe(1/time_freqs) # [B,1+2K]
        x  = self.film_flow(x, t_emb) # time-aware flow feature
        
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
            depth_3d_feature = block(depth_3d_feature, depth_3d_feature) # self attn

        ##### Ablate Flow Feature
        # return self.decoder(flow_feature.reshape(batch_size, -1))
        ##################
        return self.decoder(torch.cat((flow_feature.reshape(batch_size, -1), depth_3d_feature.reshape(batch_size, -1)), dim=1))

    
    def get_loss(self, x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs):

        pose, A = self.forward(x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs)
        
        ####### Velocity Both Trans and Rot - Testing
        # losses, pred_orth = vmf_loss_omega_k(A, y[:,6:],1/time_freqs, overreg=1.025)
        # B = time_freqs.shape[0]
        # pose[:,:3] = pose[:,:3] * (1/time_freqs).view(B,1) # translation update only
        # loss_p = torch.nn.functional.mse_loss(pose[:,:], y[:,:6])
        # loss_A = losses.mean()
        ##################

        if self.args.velocity_target:
            ######## Velocity Translation
            B = time_freqs.shape[0]
            losses, pred_orth = fisher_NLL(A, y[:,6:], overreg=1.025)
            y[:,:3] = y[:,:3] * (time_freqs).view(B,1) # augment the velocity
            y[:,:3] = symlog(y[:,:3])
            loss_p = torch.nn.functional.mse_loss(pose[:,:], y[:,:6])
            loss_A = losses.mean()
            ###############
        else:
            ######## Direct Displacement
            losses, pred_orth = fisher_NLL(A, y[:,6:], overreg=1.025)
            loss_p = torch.nn.functional.mse_loss(pose[:,:], y[:,:6])
            loss_A = losses.mean()
            ###############


        return loss_p, loss_A

    def step(self, x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs):

        loss_p, loss_A = self.get_loss(x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs)
        loss = loss_p + loss_A
        loss.backward()

        return loss_p, loss_A
