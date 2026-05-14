import re
import os
import glob
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import math
import json
import argparse
import importlib.util

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cli = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config_module", args_cli.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.args


def make_pseudo_lidar(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox)/fx
    hh = (hh.astype(np.float32) - oy)/fy
    pseudo_lidar = np.stack((ww,hh))
    return pseudo_lidar

def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    x_coords = np.arange(w).reshape(1, w)
    y_coords = np.arange(h).reshape(h, 1)
    intrinsicLayer = np.abs(x_coords - ox) / fx + np.abs(y_coords - oy) / fy
    return intrinsicLayer

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z], dtype=np.float32)

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        text = text.replace('\n\n', '\n')
        text = text.replace('\n', ' ')
        assert '\n' not in text
        sentences = re.split(r'(?<=[.!?])(?<!\b\d\.) +', text.strip())
        
    return sentences

def reduce_to_length(sentences, target_length=15):
    while len(sentences) > target_length:
        sentences.sort(key=len)
        combined_sentence = sentences[0] + " " + sentences[1]
        sentences = [combined_sentence] + sentences[2:]
    
    return sentences

def expand_to_length(sentences, target_length=15):

    while len(sentences) < target_length:
        sentences.sort(key=len, reverse=True)
        split_sentence = None
        for sentence in sentences:
            if ',' in sentence:
                split_sentence = sentence
                break

        if split_sentence:
            parts = split_sentence.split(',', 1)
            sentences.remove(split_sentence)
            sentences.extend([part.strip() for part in parts])
        else:
            break
    return sentences

def expand_to_length2(sentences, target_length=15):
    while len(sentences) < target_length:
        sentences.append(random.choice(sentences))
    return sentences

def transform_pose(poses_list):
    T = np.eye(4)
    for pose in poses_list:
        R = np.array(pose[6:]).reshape(3,3)
        t = np.array(pose[:3]).reshape(3,1)
        T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
        T_abs = np.dot(T, T_r)
        T = T_abs

    t = T[0:3, 3]
    R = T[0:3, 0:3]
    angles = rotationMatrixToEulerAngles(R)

    return np.concatenate((t, [180 / 3.1415926 * angles[0], 180 / 3.1415926 * angles[1], 180 / 3.1415926 * angles[2]], R.flatten()))

def ssl_filters(args, scene_num, geo_percent, txt_threshold, geo_filter=False, txt_filter=False):
    if txt_filter:
        f = open(args.path_prefix + f'/YouTube_json/youtube_txt_sim_dist.json') 
        txt_sim = json.load(f)
        removed_keys_txt = [key for key, value in txt_sim[scene_num].items() if value < txt_threshold]
    if geo_filter:
        f = open(args.path_prefix + f'/YouTube_json/{scene_num}_ratio.json') 
        ratio_dict = json.load(f)
        if txt_filter:
            updated_ratio_dict = {key: value for key, value in ratio_dict.items() if key not in removed_keys_txt}
        else:
            updated_ratio_dict = ratio_dict
        sorted_ratio_dict = dict(sorted(updated_ratio_dict.items(), key=lambda item: item[1]))
        top_count = int(len(sorted_ratio_dict) * geo_percent)
        selected_dict = dict(list(sorted_ratio_dict.items())[:top_count])
    return selected_dict
