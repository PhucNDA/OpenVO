import matplotlib.pyplot as plt
import numpy as np
import cv2
import pathlib
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R

import av2
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.utils.io import read_feather, read_all_annotations
from av2.evaluation.detection.eval import evaluate
from av2.evaluation.detection.utils import DetectionCfg
from av2.utils.io import read_feather, read_all_annotations
from av2.utils.synchronization_database import SynchronizationDB
import av2.utils.io as io_utils
from av2.geometry.se3 import SE3
import av2.geometry.geometry as geometry_utils

def convert_pose_dataframe_to_SE3(pose_df: pd.DataFrame) -> SE3:
    """Convert a dataframe with parameterization of a single pose, to an SE(3) object.

    Args:
        pose_df: parameterization of a single pose.

    Returns:
        SE(3) object representing the egovehicle's pose in the city frame.
    """
    qw, qx, qy, qz = pose_df[["qw", "qx", "qy", "qz"]].to_numpy().squeeze()
    tx_m, ty_m, tz_m = pose_df[["tx_m", "ty_m", "tz_m"]].to_numpy().squeeze()
    city_q_ego: NDArrayFloat = np.array([qw, qx, qy, qz])
    city_t_ego: NDArrayFloat = np.array([tx_m, ty_m, tz_m])
    city_R_ego = geometry_utils.quat_to_mat(quat_wxyz=city_q_ego)
    city_SE3_ego = SE3(rotation=city_R_ego, translation=city_t_ego)
    return city_SE3_ego

dataset_path_train= pathlib.Path('/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2_original/train')
data_train = AV2SensorDataLoader(dataset_path_train,dataset_path_train)

dataset_path_test= pathlib.Path('/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2_original/test')
data_test = AV2SensorDataLoader(dataset_path_test,dataset_path_test)

dataset_path_val= pathlib.Path('/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2_original/val')
data_val = AV2SensorDataLoader(dataset_path_val,dataset_path_val)

ids_train = data_train.get_log_ids()
ids_val = data_val.get_log_ids()
ids_test = data_test.get_log_ids()

ids = ids_train + ids_val + ids_test

print(len(ids))

city_map = {}
ids_map = {}

for id in ids_train:
    if data_train.get_city_name(id) not in city_map.keys():
        city_map[data_train.get_city_name(id)] = []

    city_map[data_train.get_city_name(id)].append(id)
    ids_map[id] = data_train.get_city_name(id)

for id in ids_test:
    if data_test.get_city_name(id) not in city_map.keys():
        city_map[data_test.get_city_name(id)] = []

    city_map[data_test.get_city_name(id)].append(id)
    ids_map[id] = data_test.get_city_name(id)

for id in ids_val:
    if data_val.get_city_name(id) not in city_map.keys():
        city_map[data_val.get_city_name(id)] = []

    city_map[data_val.get_city_name(id)].append(id)
    ids_map[id] = data_val.get_city_name(id)


# Calibration information missed
remove = [31,
 73,
 95,
 104,
 128,
 153,
 161,
 162,
 190,
 200,
 220,
 229,
 245,
 250,
 292,
 310,
 343,
 374,
 397,
 406,
 426,
 439,
 471,
 502,
 509,
 570,
 589,
 671,
 689,
 707,
 752,
 757,
 763,
 807,
 810,
 815,
 823,
 841,
 905,
 959,
 980,
 994]


camera = 'stereo_front_left'

def get_av2_intrinsics(log_root: Path, camera: str):
    """Extract fx, fy, cx, cy for a given log and camera."""
    intr_path = log_root / "calibration/intrinsics.feather"
    intr_df = io_utils.read_feather(intr_path)
    row = intr_df[intr_df["sensor_name"] == camera].iloc[0]
    fx, fy, cx, cy = row["fx_px"], row["fy_px"], row["cx_px"], row["cy_px"]
    return [float(fx), float(fy), float(cx), float(cy)]


intrinsics_dict = {}
for num, id in enumerate(ids):
    print(f'{num}/{len(ids)}')
    # if num < 850:
    #     continue
    if num in remove:
        continue
    else:
        pass
    print(num)
    if 0 <= num <= 699:
        img_folder_path = dataset_path_train / id / f"sensors/cameras/{camera}/"
        calibration = dataset_path_train / id / "calibration/egovehicle_SE3_sensor.feather"
        data = data_train
        source = dataset_path_train / id
    elif 700 <= num <= 849:
        img_folder_path = dataset_path_val / id / f"sensors/cameras/{camera}/"
        calibration = dataset_path_val / id / "calibration/egovehicle_SE3_sensor.feather"
        data = data_val
        source = dataset_path_val/ id
    elif 850 <= num <= 999:
        img_folder_path = dataset_path_test / id / f"sensors/cameras/{camera}/"
        calibration = dataset_path_test / id / "calibration/egovehicle_SE3_sensor.feather"
        data = data_test
        source = dataset_path_test / id
        
    fx, fy, cx, cy = get_av2_intrinsics(source, camera)
    intrinsics_dict[f"{str(num).zfill(3)}"] = [fx, fy, cx, cy]


    if not os.path.exists(f'/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2/20hz/{camera}/test'):
        os.makedirs(f'/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2/20hz/{camera}/test/poses')

    if not os.path.exists(f'/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2/20hz/{camera}/test/sequences/' + '{:03d}/'.format(num)):
        os.makedirs(f'/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2/20hz/{camera}/test/sequences/' + '{:03d}/'.format(num) + 'image_2/')
               
    imgs = sorted(os.listdir(img_folder_path))

    # with open(f'/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2/20hz/{camera}/test/poses/{str(num).zfill(3)}.txt', 'w') as f:
        
    #     for count, img in enumerate(imgs):
    #         # print(f'{count}/{len(imgs)}')
    #         if count % 1 == 0:
    #             pass
    #         else:
    #             continue

    #         if count == 0:
    #             location = data.get_city_SE3_ego(id,int(img[:-4]))

    #             log_poses_df = io_utils.read_feather(calibration)

    #             sub_log_poses_df = log_poses_df[log_poses_df['sensor_name'] == camera]
    #             camera_calibration = convert_pose_dataframe_to_SE3(sub_log_poses_df)
    #             Cam_M = camera_calibration.transform_matrix
    #             Cam_M_i = np.linalg.inv(camera_calibration.transform_matrix)

    #             M_init = location.transform_matrix
    #             M_init_i = np.linalg.inv(M_init)

    #         img_path = img_folder_path / img
    #         os.system('cp {} '.format(img_path) + f'/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2/20hz/{camera}/test/sequences/' + '{:03d}/'.format(num) + 'image_2/'+'{}.jpg'.format(str(count).zfill(6)))
    #         # imgs_dict[count] = img_path

    #         location = data.get_city_SE3_ego(id,int(img[:-4]))
    #         # print(location.transform_matrix)
    #         # translation.append((M_init_i@M_i@location.transform_matrix)[:3,-1])
    #         # translation.append((M_i@M_init_i@location.transform_matrix)[:3,-1])
    #         gt = Cam_M_i@M_init_i@location.transform_matrix@Cam_M
    #         gt = gt[:3, :].reshape(1,12)
    #         gt_str = str(gt[0][0]) + ' ' + str(gt[0][1]) + ' ' + str(gt[0][2]) + ' ' +\
    #                     str(gt[0][3]) + ' ' + str(gt[0][4]) + ' ' + str(gt[0][5]) + ' ' +\
    #                     str(gt[0][6]) + ' ' + str(gt[0][7]) + ' ' + str(gt[0][8]) + ' ' +\
    #                     str(gt[0][9]) + ' ' + str(gt[0][10]) + ' '+ str(gt[0][11]) + '\n'
    #         f.write(gt_str)

import json
with open(f'/fs/nexus-projects/AD_dashrecon/ADVO/data/Argoverse_2/20hz/{camera}/test/av2_est_intrs.json', "w") as f:
    json.dump(intrinsics_dict, f, indent=2)

