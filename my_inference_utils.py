import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2
import json
import random
import matplotlib
import itertools
matplotlib.use('Agg')

# Model
from model import *
from fisher.fisher_utils import vmf_loss as fisher_NLL, fisher_CE, batch_torch_A_to_R, fisher_entropy
from fisher.fisher_utils import vmf_loss_omega_k
from utils import *
from utils import read_config
# Dataset
from torch.utils.data import Dataset, DataLoader
from visualization.visualizer import visualizer, show_pcd
from dataset import get_infer_data_info, InferDataset
from argparse import Namespace

def _strip_module_prefix(state):
    new = OrderedDict()
    for k, v in state.items():
        new[k[7:] if k.startswith("module.") else k] = v
    return new

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Visual Odometry")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser


s = torch.tensor([3.6]).cuda()
y_max = torch.tensor([75.0]).cuda()  # m/s
alpha = torch.log(1 + s * abs(y_max))
def symlog(y):
    return torch.sign(y) * torch.log(1 + s * torch.abs(y)) / alpha
def inv_symlog(t):
    return torch.sign(t) * (torch.exp(torch.abs(t) * alpha) - 1) / s


def inv_symlog(t):
    return torch.sign(t) * (torch.exp(torch.abs(t) * alpha) - 1) / s


class InferenceVO_Util:
    def __init__(self, args, root_path, save_path, weight, scheduler):
        self.root_path = root_path
        self.save_path = save_path # '../results'
        self.weight = weight
        self.args = args
        if args.model_type == 'openvo':
            self.model = OpenVO(args)
        elif args.model_type == 'zvo_lite':
            self.model = ZVOModel(args)
        # self.model = VOModel(self.args)

        self.model = self.model.to('cuda')
        
        self.inference_data = self.args.inference_data
        self.scheduler = scheduler

        # Load Pretrained
        pth = weight
        epoch = pth.split('.')[0].split('-')[-1]

        print('Restored model from epoch {}:'.format(epoch))
        checkpoint = torch.load(pth, map_location="cpu")
        state = checkpoint["model_state_dict"]
        if not hasattr(self.model, "module"):
            state = _strip_module_prefix(state)
        self.model.to('cuda')
        self.model.load_state_dict(state)
        self.model.eval()
    
    def runVO(self, start, end):
        for idx in range(start, end , 1):
            scene_info, data_path, depth_path, intrinsics = self.scheduler[idx][0], self.scheduler[idx][1], self.scheduler[idx][2], self.scheduler[idx][3]
            
            key = next(iter(scene_info))
            scene = scene_info[key][0]
            # KITTI: 10HZ
            # NUSC: 12HZ
            # ARGO = 10 || 20
            time_freq = 0
            if 'KITTI' in scene_info:
                time_freq = 10
            elif 'NUSC' in scene_info:
                time_freq = 12
            elif 'ARGO' in scene_info:
                time_freq = 20 #|| 10 based on your sampling rate
            else:
                time_freq = 10 # else

            df = get_infer_data_info(self.args, scene_info, data_path, depth_path, intrinsics, time_freq, None)
            dataset = InferDataset(df, (self.args.img_h, self.args.img_w))
            dataloader = DataLoader(
                dataset, 
                batch_size=16, # set = 1 for debugging
                shuffle=False, 
                num_workers=self.args.n_processors,
                pin_memory=True,
            )
            poses_dict = []
            scene_poses = [[1.0,0.0,0.0,0.0,
                        0.0,1.0,0.0,0.0,
                        0.0,0.0,1.0,0.0]]
            pose_it = 0
            with torch.no_grad():
                for step, (img_paths, x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs, scene_pose) in enumerate(tqdm(dataloader)):
                    x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs = x.to('cuda'), intrs.to('cuda'), intrs_map.to('cuda'), depth_map0.to('cuda'), depth_map1.to('cuda'), depth_3d.to('cuda'), text_features.to('cuda'), time_freqs.to('cuda')
                    # Show PointCloud for debugging
                    # show_pcd(img = x[0,:3,:,:], lidar_xy = depth_3d[0,2:,:,:], depth = depth_map0[0], save_path = '../debug/')
                    predicted_p, predicted_r = self.model.forward(x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, text_features, time_freqs)
                    
                    if self.args.velocity_target:
                        #### Translation Velocity
                        pred_orth = fisher_NLL(predicted_r, None, overreg=1.025)
                        B = time_freqs.shape[0]
                        predicted_p[:,:3] = inv_symlog(predicted_p[:,:3]) * (1.0/(time_freqs).view(B,1))
                    else:
                        #### Direct Displacement
                        pred_orth = fisher_NLL(predicted_r, None, overreg=1.025)
                        
                    #### Velocity
                    # B = time_freqs.shape[0]
                    # predicted_p[:,:3] = predicted_p[:,:3] * (1/time_freqs).view(B,1)
                    # pred_orth = vmf_loss_omega_k(predicted_r, None, 1/time_freqs, overreg=1.025)
                    ##################

                    predicted_p = predicted_p.view(-1,6).data.cpu().numpy()
                    pred_orth = pred_orth.data.cpu().numpy()
                    for i in range(len(predicted_p)):
                        poses_dict.append(list(predicted_p[i]) + list(pred_orth[i].flatten()))
                        scene_poses.append(0)
                        pose_it+=1

            abs_est = [[1.0,0.0,0.0,0.0,
                        0.0,1.0,0.0,0.0,
                        0.0,0.0,1.0,0.0]]
            T = np.eye(4)
            for _pose in poses_dict:

                R = np.array(_pose[6:]).reshape(3, 3)
                t = np.array(_pose[:3]).reshape(3, 1)
                T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
                T_abs = np.dot(T, T_r)
                T = T_abs
                abs_est.append(T[0:3, :].flatten().tolist())
            
            with open('{}/{}/{}/{}/{}.txt'.format(self.save_path, key, self.weight.split('/')[-2], self.weight.split('/')[-1].split('.')[0], scene), 'w') as f:
                for pose in abs_est:
                    f.write(' '.join(str(e) for e in pose))
                    f.write('\n')
            ## Online Visualization
            # visualizer(self.inference_data, self.save_path, self.weight)

        

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Everything from the InferenceVO instance is in cfg
    root_path     = cfg["root_path"]
    save_path     = cfg["save_path"]
    weight        = cfg["weight"]
    scheduler     = cfg["scheduler"]
    args_dict     = Namespace(**cfg["args"])    
    worker_rank   = cfg["worker"]["rank"]
    start, end    = cfg["worker"]["range"]

    IVO = InferenceVO_Util(args_dict, root_path, save_path, weight, scheduler)
    IVO.runVO(start, end)
    