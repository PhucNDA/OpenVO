import os
import glob
import torch
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
from utils import *
from utils import read_config
# Dataset
from torch.utils.data import Dataset, DataLoader
from visualization.visualizer import visualizer, show_pcd

# Process
import os, json, itertools, subprocess, tempfile
from types import SimpleNamespace

class InferenceVO:
    def __init__(self, root_path, save_path, weight, nprocess):
        self.root_path = root_path
        self.save_path = save_path # './results'
        self.weight = weight
        self.args = read_config()
        self.inference_data = self.args.inference_data
        
        
        # Make Dirs
        self.already = True
        for key in self.inference_data.keys():
            os.makedirs('{}/{}/{}/{}'.format(self.save_path, key, self.weight.split('/')[-2], self.weight.split('/')[-1].split('.')[0]), exist_ok=True)
            os.makedirs('{}/{}/{}/{}'.format(self.save_path, key, self.weight.split('/')[-2], self.weight.split('/')[-1].split('.')[0]+'_gt'), exist_ok=True)
            self.inference_data[key][key] = sorted(self.inference_data[key][key])

        # Check Not Exists
        import copy
        self.temporary = copy.deepcopy(self.inference_data)
        for key in self.inference_data.keys():
            idxes = []
            for idx, scene_id in enumerate(self.inference_data[key][key]):
                if not os.path.exists('{}/{}/{}/{}/{}'.format(self.save_path, key, self.weight.split('/')[-2], self.weight.split('/')[-1].split('.')[0], scene_id.zfill(3) + '.txt')) and not os.path.exists('{}/{}/{}/{}/{}'.format(self.save_path, key, self.weight.split('/')[-2], self.weight.split('/')[-1].split('.')[0], scene_id.zfill(2) + '.txt')):
                    self.already = False
                    idxes.append(idx)
            self.temporary[key][key] = []
            # fill remaining
            for idx in idxes:
                self.temporary[key][key].append(self.inference_data[key][key][idx])
        
        self.inference_data = copy.deepcopy(self.temporary)
        # Job Scheduling based on nprocess & inference_data
        self.scheduler = []
        for map, value in self.inference_data.items():
            for key, scenes in value.items():
                for s in scenes:
                    self.scheduler.append(({key: [s]}, self.args.data_path, self.args.depth_path, self.args.data_intrinsics))
        total_action = len(self.scheduler)
        budget = total_action // nprocess
        remainder = total_action % nprocess
        self.ranges = [(s, e) for s, e in zip([0]+list(itertools.accumulate([budget + (i<remainder) for i in range(nprocess)])), itertools.accumulate([budget + (i<remainder) for i in range(nprocess)]))]

    def _to_serializable(self, x):
        if hasattr(x, "__dict__"):
            try:
                return {k: self._to_serializable(v) for k, v in x.__dict__.items()}
            except Exception:
                return str(x)
        if isinstance(x, (list, tuple)):
            return [self._to_serializable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): self._to_serializable(v) for k, v in x.items()}
        return x 

    def _pack_attributes(self):
        d = dict(self.__dict__)
        if "args" in d:
            a = d["args"]
            if hasattr(a, "__dict__"):
                d["args"] = {k: self._to_serializable(v) for k, v in a.__dict__.items()}
        return self._to_serializable(d)

    def launch_workers(self, run_py="my_inference_utils.py", devices=None, python_bin="python"):
        nprocess = len(self.ranges)
        if devices is None:
            devices = list(range(nprocess))
        if len(devices) < nprocess:
            q, r = divmod(nprocess, len(devices))
            devices = (devices * q) + devices[:r]

        os.makedirs(self.save_path, exist_ok=True)

        base_payload = self._pack_attributes()
        procs = []
        json_paths = []
        for rank, ((start, end), dev) in enumerate(zip(self.ranges, devices)):
            payload = dict(base_payload)
            payload["worker"] = {"rank": rank, "range": [start, end]}
            # write a per-worker config
            cfg_path = os.path.join(
                self.save_path,
                f"worker_cfg_rank{rank}_dev{dev}.json"
            )
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(dev)

            # Launch non-blocking (background) just like " & " in bash
            cmd = [python_bin, run_py, "--config", cfg_path]
            procs.append(subprocess.Popen(cmd, env=env))
            json_paths.append(cfg_path)
        return procs, json_paths
    


if __name__ == '__main__':
    root = './data'
    weights = './weights/OpenVO/openvo_nusc_gt'
    save_path = "./results_test"
    
    
    paths = os.listdir(weights)
    ## Manual
    # paths = ['model_ep-022.pt']
    
    for path in paths:
        if not path.endswith('.pt'):
            continue
        print(f'------- {path} --------')
        weight = os.path.join(weights, path)
        ivo = InferenceVO(root_path=root, save_path=save_path, weight=weight, nprocess=1) # 4
        if ivo.already == True: # already evaluated!
            continue
        procs, json_pths = ivo.launch_workers(run_py="src/my_inference_utils.py", devices=[0])  # returns Popen handles 4 || 0->(0,1,2,3)
        for p in procs:
            p.wait() # wait till finished!
        ## Visualize # toggle on
        # visualizer(ivo.inference_data, save_path, weight)
