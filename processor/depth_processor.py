
import os
import cv2
import torch
import json
import random
import numpy as np
from PIL import Image
try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction
import sys
sys.path.append(os.path.join(os.environ["PWD"], "processor", "Metric3D"))
from mono.model.monodepth_model import get_configured_monodepth_model
sys.path.append(os.path.join(os.environ["PWD"], "processor", "WildCamera"))
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from tqdm import tqdm


def set_seed(seed: int = 42, deterministic: bool = True):
    # Python & OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    if deterministic:
        # Make cuDNN & PyTorch pick deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # (faster, but may be non-deterministic)
        torch.backends.cudnn.benchmark = True
    # Optional (PyTorch 2.x): control matmul numerics
    try:
        torch.set_float32_matmul_precision("high")  # or "medium"/"highest"
    except Exception:
        pass

set_seed(1234) # fixed seed
metric3d_dir = './processor/Metric3D'
MODEL_TYPE = {
  'ConvNeXt-Tiny': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
  },
  'ConvNeXt-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
  },
  'ViT-Small': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
  },
  'ViT-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
  },
  'ViT-giant2': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
  },
}


def metric3d_convnext_tiny(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
    pretrain (bool): whether to load pretrained weights.
    Returns:
    model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
        )
    return model

def metric3d_convnext_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
    pretrain (bool): whether to load pretrained weights.
    Returns:
    model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
        )
    return model

def metric3d_vit_small(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
    pretrain (bool): whether to load pretrained weights.
    Returns:
    model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
        )
    return model

def metric3d_vit_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
    pretrain (bool): whether to load pretrained weights.
    Returns:
    model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
        )
    return model

def metric3d_vit_giant2(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
    pretrain (bool): whether to load pretrained weights.
    Returns:
    model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
        )
    return model

def save_depth_png(depth_m, path_png, invalid_zero=True):
    """
    Save metric depth (meters) to KITTI-style 16-bit PNG:
        uint16_value = round(depth_m * 256)
        0 reserved for invalid/no-depth pixels.
        img = cv2.imread(..., cv2.IMREAD_UNCHANGED)
            Eg: uint16 ~ 1206 44223
    """
    # to numpy HxW float32
    if isinstance(depth_m, torch.Tensor):
        depth_m = depth_m.detach().cpu().squeeze().float().numpy()
    d = np.asarray(depth_m, dtype=np.float32)

    # validity mask (you can customize)
    valid = np.isfinite(d) & (d > 0)

    # scale meters -> KITTI units and convert
    d16 = np.round(d * 256.0)
    d16 = np.clip(d16, 0, 65535).astype(np.uint16)

    # mark invalid as 0 (or 65535 if you prefer a sentinel)
    d16[~valid] = 0 if invalid_zero else 65535

    # write as 16-bit PNG (no normalization)
    cv2.imwrite(path_png, d16)

class Depth_Processor:
    def __init__(self, root = None):
        # NeWCRFs model
        self.intrinsic_model = NEWCRFIF(version='large07', pretrained=None)
        self.intrinsic_model.eval()
        self.intrinsic_model.cuda()
        # Load Intrinsic Model
        ckpt_path_intrinsic = './weights/WildCamera/wild_camera_all.pth'
        self.intrinsic_model.load_state_dict(torch.load(ckpt_path_intrinsic, map_location="cpu"), strict=True)
        # Load Depth Model
        ckpt_path_depth = './weights/Metric3D/metric_depth_vit_large_800k.pth'
        self.depth_model = metric3d_vit_large()
        self.depth_model.load_state_dict(torch.load(ckpt_path_depth)['model_state_dict'], strict=False)
        self.depth_model.cuda().eval()
        
        # Following KITTI
        self.root = root
        self.scene_ids = sorted(os.listdir(os.path.join(self.root,'sequences')))
        self.n_scene = len(self.scene_ids)

    def process(self, start_id, end_id, use_gt = None):
        '''
        Process all scene
        '''
        import time
        intrinsics = {}
        for idx, scene_id in enumerate(self.scene_ids[start_id:end_id]):
            print(f'-----------------{idx}/{self.n_scene}-----------------')
            start_time = time.time()
            images_folder = os.path.join(self.root, 'sequences', scene_id, 'image_2')
            if not use_gt:
                depths_folder = os.path.join(self.root, 'depth_est_intrs', 'sequences', scene_id, 'image_2')
            else:
                depths_folder = os.path.join(self.root, 'depth_gt_intrs', 'sequences', scene_id, 'image_2')

            
            if not use_gt:
                intrinsic = self.process_intrinsic(images_folder)
                intrinsics[scene_id] = intrinsic
            else:
                with open(use_gt, 'r') as f:
                    intrinsics_dict = json.load(f)
                intrinsic = intrinsics_dict[scene_id]
            
            if not os.path.exists(depths_folder):
                os.makedirs(depths_folder, exist_ok=True)
                depth_est = self.process_depth(images_folder, intrinsic, depths_folder)

            end_time = time.time()
            elapsed_sec = end_time - start_time
            minutes, seconds = divmod(int(elapsed_sec), 60)
            print(f"Time taken: {minutes:02d}:{seconds:02d}")
        if not use_gt:
            with open(os.path.join(self.root, f'est_intrs_{start_id}-{end_id}.json'), "w") as f:
                json.dump(intrinsics, f, indent=2)
        else:
            pass

    def process_intrinsic(self, images_folder):
        '''
        Process Intrinsic
        '''
        sumfx = 0
        sumfy = 0
        sumcx = 0
        sumcy = 0
        cnt = 0
 
        for idx, info in enumerate(tqdm(sorted(os.listdir(images_folder)))):
            image_path = os.path.join(images_folder, info)
            intrinsic, _ = self.intrinsic_model.inference(Image.open(image_path), wtassumption=False)
            focal = intrinsic[0, 0].item()

            fx = intrinsic[0,0].item()
            fy = intrinsic[1,1].item() 
            cx = intrinsic[0,2].item() 
            cy = intrinsic[1,2].item()
            sumfx += fx
            sumfy += fy
            sumcx += cx
            sumcy += cy
            cnt += 1
        return [sumfx/cnt, sumfy/cnt, sumcx/cnt, sumcy/cnt]


    def process_depth(self, images_folder, intrinsic, depths_folder):
        '''
        Process Depth
        '''
        for idx, info in enumerate(tqdm(sorted(os.listdir(images_folder)))):
            images_path = os.path.join(images_folder, info)
            # read & RGB
            rgb_origin = cv2.imread(images_path)[:, :, ::-1]
            H0, W0 = rgb_origin.shape[:2]

            # ---- resize keeping ratio (for ViT)

            # input_size = (775, 1024)  # (H, W) # 
            # input_size = (616, 1064)  # (H, W) # AV2
            input_size = (1064, 616)  # (H, W) # Nusc || KITTI
            
            scale = min(input_size[0] / H0, input_size[1] / W0)
            rgb = cv2.resize(rgb_origin, (int(W0 * scale), int(H0 * scale)), cv2.INTER_LINEAR)

            # intrinsics vector [fx, fy, cx, cy] at original resolution
            fx, fy, cx, cy = intrinsic

            # after resize (padding doesn't change focal)
            fx_net = fx * scale
            fy_net = fy * scale

            # pad to model input
            pad_h = input_size[0] - rgb.shape[0]
            pad_w = input_size[1] - rgb.shape[1]
            pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                                    pad_w_half, pad_w - pad_w_half,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[123.675,116.28,103.53])

            # if you need cx, cy at network input later:
            cx_net = cx * scale + pad_w_half
            cy_net = cy * scale + pad_h_half

            # normalize (RGB, ImageNet stats)
            mean = torch.tensor([123.675,116.28,103.53]).view(3,1,1).float()
            std  = torch.tensor([58.395,57.12,57.375]).view(3,1,1).float()
            rgb = torch.from_numpy(rgb.transpose(2,0,1)).float()
            rgb = (rgb - mean) / std
            inp = rgb.unsqueeze(0).cuda()

            with torch.no_grad():
                pred_depth, confidence, output_dict = self.depth_model.inference({'input': inp})
            # pred_depth: HxW at model input (canonical-depth)

            # unpad to resized canvas
            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[pad_h_half:pred_depth.shape[0] - (pad_h - pad_h_half),
                                    pad_w_half:pred_depth.shape[1] - (pad_w - pad_w_half)]

            # upsample back to original resolution (bilinear is OK for depth)
            pred_depth = torch.nn.functional.interpolate(
                pred_depth[None, None], size=(H0, W0), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)

            # ---- canonical -> metric
            canonical_f = 1000.0
            scale_metric = fx_net / canonical_f        # use fx after resize
            pred_depth_metric = torch.clamp(pred_depth * scale_metric, 0, 300)
            
            destination = os.path.join(depths_folder, str(idx).zfill(6) + '.png')
            save_depth_png(pred_depth_metric, destination)


# ---------------------------------------------------------------------- MAIN FUNCTION # ---------------------------------------------------------------------- 

datapath = 'data/nuScenes/NUSCv1.0-trainval_12hz/CAM_FRONT'

proc = Depth_Processor(root = datapath)
proc.process(0,100, use_gt = None) # Use start_id, end_id if you have multiple threads 

# For training datasets, toggle the use_gt='instrinsics-of-respective-datasets.json' to generate reliable depth for training
# For testing, we will be using estimated depth for paper benchmarking
# ------------------------------------------------------------------------------------------------------------------------------------------------------------- 