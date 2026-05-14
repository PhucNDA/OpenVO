from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from pyquaternion import Quaternion
import pandas as pd
from tqdm import tqdm, trange

import math
from scipy.spatial.transform import Rotation as R
import json

class NUSC_Process:
    def __init__(self, root, version='v1.0-mini'):
        # nusc Mini
        self.nusc = NuScenes(version=version, dataroot=root)        
        self.root = root
        self.create_mapscene()

    def create_mapscene(self):
    # --------------------- Create Mapscene ---------------------
        self.map_scene = {}
        self.total_scene = len(self.nusc.scene)
        for i in range(self.total_scene):
            if self.nusc.get('log', self.nusc.scene[i]['log_token'])['location'] not in self.map_scene.keys():
                self.map_scene[self.nusc.get('log', self.nusc.scene[i]['log_token'])['location']] = []
            self.map_scene[self.nusc.get('log', self.nusc.scene[i]['log_token'])['location']].append(str(i).zfill(3))


    def sequence12hz(self, image_path):
    # --------------------- Generate 12Hz Version Scene ---------------------
        # Note: Lidar 20Hz, Camera: 12Hz
        for i in tqdm(range(self.total_scene)):
            if not os.path.exists(image_path + '{:03d}/'.format(i)):
                os.makedirs(image_path + '{:03d}/'.format(i), exist_ok=True)
                os.makedirs(image_path + '{:03d}/'.format(i) + 'image_2/', exist_ok=True)
                
            first_sample = self.nusc.get('sample', self.nusc.scene[i]['first_sample_token'])
            sample_data = self.nusc.get('sample_data', first_sample['data']['CAM_FRONT'])
            num = 0
            while True:
                os.system(f'cp {self.root}{sample_data["filename"]} {image_path}/{i:03d}/image_2/{num:06d}.jpg')
                num = num + 1
                if sample_data['next'] != '':
                    sample_data = self.nusc.get('sample_data', sample_data['next'])
                else:
                    break

    def dump_intrinsics_json(self, out_json: str, cam_channel: str = "CAM_FRONT"):
    # --------------------- intrinsic 12Hz Version Scene ---------------------
        """
        Save per-scene native intrinsics as:
        {
          "000": [fx, fy, cx, cy],
          "001": [fx, fy, cx, cy],
          ...
        }
        """
        data = {}
        for scene_idx in trange(self.total_scene):
            # first frame of the scene for this camera
            sample = self.nusc.get('sample', self.nusc.scene[scene_idx]['first_sample_token'])
            sd = self.nusc.get('sample_data', sample['data'][cam_channel])
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

            K = np.array(cs['camera_intrinsic'], dtype=float)  # 3x3, native size
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

            data[f"{scene_idx:03d}"] = [fx, fy, cx, cy]

        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)


    def egopose12hz(self, pose_path):
    # --------------------- Pose 12Hz Version Scene ---------------------
        # Note: Lidar 20Hz, Camera: 12Hz
        os.makedirs(pose_path, exist_ok=True)    
        for num in range(self.total_scene):

            current_scene = self.nusc.scene[num]
            sample_token = current_scene['first_sample_token']
            sample = self.nusc.get('sample', sample_token)
            cam_token = sample['data']['CAM_FRONT']
            cam = self.nusc.get('sample_data', cam_token)

            first_pose = self.nusc.get('ego_pose', cam['ego_pose_token'])
            first_pose_mat = transform_matrix(first_pose['translation'], Quaternion(first_pose['rotation']), inverse=False)

            first_calibrated_sensor_token = cam['calibrated_sensor_token']
            _first_calibrate_sensor = self.nusc.get('calibrated_sensor', first_calibrated_sensor_token)
            first_cam_pose_mat = transform_matrix(_first_calibrate_sensor['translation'], Quaternion(_first_calibrate_sensor['rotation']), inverse=False)

            cords = np.zeros((1, 4))
            cords[:,-1] = 1

            x = []
            y = []

            with open(f'{pose_path}{num:03d}.txt', 'w') as f:
                while True:
                    cam = self.nusc.get('sample_data', cam_token)
                    calibrated_sensor_token = cam['calibrated_sensor_token']

                    pose = self.nusc.get('ego_pose', cam['ego_pose_token'])
                    cur_pose_mat = transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=False)

                    _calibrate_sensor = self.nusc.get('calibrated_sensor', calibrated_sensor_token)
                    cur_cam_pose_mat = transform_matrix(_calibrate_sensor['translation'], Quaternion(_calibrate_sensor['rotation']), inverse=False)
                    
                    _temp_pose = np.dot(np.linalg.inv(first_cam_pose_mat), np.linalg.inv(first_pose_mat))
                    _new_pose = np.dot(_temp_pose, cur_pose_mat)
                    _new_pose = np.dot(_new_pose, cur_cam_pose_mat)

                    _rotation = _new_pose[:3, :3]

                    _loc = np.dot(_new_pose, np.transpose(cords))

                    x.append(_loc[0][0])
                    y.append(_loc[2][0])

                    f.write(str(_rotation[0][0]) + ' ')
                    f.write(str(_rotation[0][1]) + ' ')
                    f.write(str(_rotation[0][2]) + ' ')

                    f.write(str(_loc[0][0]) + ' ')

                    f.write(str(_rotation[1][0]) + ' ')
                    f.write(str(_rotation[1][1]) + ' ')
                    f.write(str(_rotation[1][2]) + ' ')

                    f.write(str(_loc[1][0]) + ' ')

                    f.write(str(_rotation[2][0]) + ' ')
                    f.write(str(_rotation[2][1]) + ' ')
                    f.write(str(_rotation[2][2]) + ' ')

                    f.write(str(_loc[2][0]) + '\n')

                    cam_token = cam['next']

                    if cam_token == '':
                        break



# ---------------------------------------------------------------------- MAIN FUNCTION # ---------------------------------------------------------------------- 
datapath = '/fs/nexus-projects/AD_dashrecon/ADVO/OpenVO' # Please indicate the location of OpenVO directory
root = '/fs/gamma-datasets/nuscenes/' # Please indicate location of database
version = 'v1.0-trainval'
# version = 'v1.0-mini'

nusc = NUSC_Process(root, version)
nusc.dump_intrinsics_json(f'{datapath}/data/nuScenes/NUSC{version}_12hz/CAM_FRONT/nuscene_gt_intrs.json')
nusc.sequence12hz(f'{datapath}/data/nuScenes/NUSC{version}_12hz/CAM_FRONT/sequences/')
nusc.egopose12hz(f'{datapath}/data/nuScenes/NUSC{version}_12hz/CAM_FRONT/poses/')
# ------------------------------------------------------------------------------------------------------------------------------------------------------------- 