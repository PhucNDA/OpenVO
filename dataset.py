import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
import random
import cv2
import math
import re
from utils import *
from torchvision.transforms.functional import resize as tv_resize, InterpolationMode

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

def matrix_rt(p):
    return np.vstack([np.reshape(p.astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])

def create_pose_data(args,):
    if not os.path.exists(f'./poses'):
        os.mkdir(f'./poses')

    for key in args.data_path.keys():
        if not os.path.exists(f'./poses/{key}'):
            os.mkdir(f'./poses/{key}')
        
        poses_dir = glob.glob(args.data_path[key] + '/poses/*.txt')
        
        for dir in poses_dir:
            with open(dir) as f:
                raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
            
            poses = []
            for i in range(len(raw_poses)-1):
                pose1 = matrix_rt(raw_poses[i])
                pose2 = matrix_rt(raw_poses[i + 1])
                pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
                R = pose2wrt1[0:3, 0:3]
                t = pose2wrt1[0:3, 3]
                angles = rotationMatrixToEulerAngles(R)
                poses.append(np.concatenate((t, [180 / 3.1415926 * angles[0], 180 / 3.1415926 * angles[1], 180 / 3.1415926 * angles[2]], R.flatten()))) #Degree
            
            poses = np.array(poses)
            np.save(f'./poses/{key}/' + dir.split('/')[-1][:-4] +'.npy', poses)

def create_youtube_pose_data(pose_path):

    if not os.path.exists(f'./poses/YouTube'):
        os.mkdir(f'./poses/YouTube')
    
    poses_dir = glob.glob(pose_path + '/*.txt')
    
    for dir in poses_dir:
        with open(dir) as f:
            raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
        
        poses = []
        for i in range(len(raw_poses)-1):
            pose1 = matrix_rt(raw_poses[i])
            pose2 = matrix_rt(raw_poses[i + 1])
            pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
            R = pose2wrt1[0:3, 0:3]
            t = pose2wrt1[0:3, 3]
            angles = rotationMatrixToEulerAngles(R)
            poses.append(np.concatenate((t, [180 / 3.1415926 * angles[0], 180 / 3.1415926 * angles[1], 180 / 3.1415926 * angles[2]], R.flatten()))) #Degree
        
        poses = np.array(poses)
        np.save(f'./poses/YouTube/' + dir.split('/')[-1][:-4] +'.npy', poses)
########################################################################################################
#                                          Training Dataset                                            #
########################################################################################################

def get_data_info(training_data, args, mode):
    '''
    Args:
        data_training (dictionary): args.data_training
        data_pth (dictionary): args.data_pth

    Return:
        DataFrame: ['imgs_paths', 'poses']
    '''
    imgs_paths = []
    poses = []
    intrinsics = []
    depth_map_paths = []
    flip_marks = []

    time_freqs = []
    time_banks = args.adaptiveHZ

    examples_num = 0
    rot_examples_num = 0
    large_rot_examples_num = 0
    
    removed_indices_num = 0

    for time_freq in time_banks:
        for key in training_data.keys():
            # print(key)
            for scene in training_data[key]:
                # print(scene)
                if key == "YouTube":
                    ytb_rot_examples_num = 0
                    ytb_straight_examples_num = 0
                    ytb_rot_remove_num = 0
                    ytb_straight_remove_num = 0
                    selected_dict = ssl_filters(args, scene, geo_percent=0.5, txt_threshold=5, geo_filter=True, txt_filter=True)

                # scene_poses = list(np.load(f'./poses/{key}/{scene}.npy')) # Array: [N-1, 15]
                
                ############# Edit HZ
                time_step = 12 // time_freq
                scene_poses = []
                scene_freqs = []
                pose_path = os.path.join(args.data_path[key], 'poses', f'{scene}.txt' )
                with open(pose_path) as f:
                    raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                for i in range(len(raw_poses)-time_step):
                    pose1 = matrix_rt(raw_poses[i])
                    pose2 = matrix_rt(raw_poses[i + time_step])
                    pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
                    R = pose2wrt1[0:3, 0:3]
                    t = pose2wrt1[0:3, 3]
                    angles = rotationMatrixToEulerAngles(R)
                    scene_poses.append(np.concatenate((t, [180 / 3.1415926 * angles[0], 180 / 3.1415926 * angles[1], 180 / 3.1415926 * angles[2]], R.flatten()))) #Degree
                    scene_freqs.append(time_freq)
                

                scene_poses = list(np.array(scene_poses))            
                #############

                scene_imgs_path = glob.glob(f'{args.data_path[key]}/sequences/{scene}/image_2/*.jpg') + glob.glob(f'{args.data_path[key]}/sequences/{scene}/image_2/*.png')
                scene_imgs_path = sorted(scene_imgs_path) # List: N
                scene_imgs_path_paired = [[scene_imgs_path[i], scene_imgs_path[i + time_step]] for i in range(len(scene_imgs_path) - time_step)] # List: N-1
                assert len(scene_imgs_path_paired) == len(scene_poses)

                scene_depth_path = glob.glob(f'{args.depth_path[key]}/sequences/{scene}/image_2/*.png')
                scene_depth_path = sorted(scene_depth_path) # List: N
                scene_depth_path_paired = [[scene_depth_path[i], scene_depth_path[i + time_step]] for i in range(len(scene_depth_path) - time_step)] # List: N-1
                assert len(scene_depth_path_paired) == len(scene_poses)

                _intrinsics = [args.data_intrinsics[key][scene]] * len(scene_poses)

                _flip_marks = [False] * len(scene_poses)

                if mode == 'train' and key == "YouTube":
                    index_removed = []
                    for i, _pose in enumerate(scene_poses):
                        if abs(_pose[4]) > 0.3:
                            ytb_rot_examples_num = ytb_rot_examples_num + 1
                            if str(i // 10 * 10).zfill(6) in selected_dict.keys():
                                pass
                            else:
                                ytb_rot_remove_num = ytb_rot_remove_num + 1
                                index_removed.append(i)
                        else:
                            ytb_straight_examples_num = ytb_straight_examples_num + 1
                            if str(i // 10 * 10).zfill(6) in selected_dict.keys():
                                if random.choice([True, True, True, True, True, True, True, True, True, False]):
                                    ytb_straight_remove_num = ytb_straight_remove_num + 1
                                    index_removed.append(i)
                            else:
                                ytb_straight_remove_num = ytb_straight_remove_num + 1
                                index_removed.append(i)
                    index_removed.reverse()
                    for i in index_removed:
                        del scene_poses[i]
                        del scene_imgs_path_paired[i]
                        del scene_depth_path_paired[i]
                        del scene_freqs[i]
                        del _intrinsics[i]
                        del _flip_marks[i]
                    removed_indices_num = removed_indices_num + len(index_removed)

                    rotation_percentage = ((ytb_rot_examples_num-ytb_rot_remove_num) / ytb_rot_examples_num) * 100
                    straight_percentage = ((ytb_straight_examples_num-ytb_straight_remove_num) / ytb_straight_examples_num) * 100
                    print(f"YouTube {scene} has {ytb_rot_examples_num} rotation examples and {ytb_straight_examples_num} straight examples. {ytb_rot_examples_num-ytb_rot_remove_num}, {rotation_percentage:.2f}% rotation examples are selected, {ytb_straight_examples_num-ytb_straight_remove_num}, {straight_percentage:.2f}% straight examples are selected.")

                imgs_paths = imgs_paths + scene_imgs_path_paired
                poses = poses + scene_poses
                intrinsics = intrinsics + _intrinsics
                depth_map_paths = depth_map_paths + scene_depth_path_paired
                time_freqs = time_freqs + scene_freqs
                flip_marks = flip_marks + _flip_marks

                if args.data_flip and mode == 'train' and key != "YouTube":
                    for i, _pose in enumerate(scene_poses):
                        examples_num = examples_num + 1

                        if abs(_pose[4]) > 0.3 and abs(_pose[4]) <= 1:
                            rot_examples_num = rot_examples_num + 1

                            imgs_paths = imgs_paths + [scene_imgs_path_paired[i]]*args.k
                            poses = poses + [_pose]*args.k
                            intrinsics = intrinsics + [_intrinsics[i]]*args.k
                            depth_map_paths = depth_map_paths + [scene_depth_path_paired[i]]*args.k
                            time_freqs = time_freqs + [time_freqs[i]]*args.k
                            flip_marks = flip_marks + [True]*args.k
                        elif abs(_pose[4]) > 1:
                            large_rot_examples_num = large_rot_examples_num + 1

                            imgs_paths = imgs_paths + [scene_imgs_path_paired[i]]*(args.k+1)
                            poses = poses + [_pose]*(args.k+1)
                            intrinsics = intrinsics + [_intrinsics[i]]*(args.k+1)
                            depth_map_paths = depth_map_paths + [scene_depth_path_paired[i]]*(args.k+1)
                            time_freqs = time_freqs + [time_freqs[i]]*(args.k + 1)
                            flip_marks = flip_marks + [False if i % 2 != 0 else True for i in range(args.k+1)]


    assert len(imgs_paths) == len(poses)

    if args.data_flip and mode == 'train':
        print("Driving straight examples number:", examples_num-rot_examples_num-large_rot_examples_num, "Regular rotation examples number:", rot_examples_num, "Large rotation examples number:", large_rot_examples_num, "Total examples number:", examples_num+rot_examples_num*args.k+large_rot_examples_num*(args.k+1))
    data = {'imgs_paths': imgs_paths, 'poses': poses, 'intrinsics': intrinsics, 'depth_paths': depth_map_paths, 'flip_marks': flip_marks, 'time_freqs': time_freqs}
    df = pd.DataFrame(data, columns = ['imgs_paths', 'poses', 'intrinsics', 'depth_paths', 'flip_marks', 'time_freqs'])

    return df

def read_depth(path):
    d = Image.open(path)
    return d

class VisualOdometryDataset(Dataset):
    def __init__(self, args, data_df, img_size, mode):
        """
        Args:
            data_df: with columns imgs_paths, poses, intrinsics, depth_paths, flip_marks
            img_size: (H, W)
        Returns from __getitem__ (test):
            (path0, imgs[6xHxW], poses_gt[...], intrs[4], intrinsicLayer[CxHxW], depth0[1xHxW], depth1[1xHxW], depth_3d[(1+C+2)xHxW])
        """
        self.args = args
        self.data_df = data_df
        self.imgs_paths = np.asarray(data_df.imgs_paths)
        self.poses = np.asarray(data_df.poses)
        self.img_size = img_size  # (H, W)
        self.intrinsics = np.asarray(data_df.intrinsics)
        self.depth_paths = np.asarray(data_df.depth_paths)
        self.flip_marks = data_df.flip_marks
        self.time_freqs = data_df.time_freqs
        self.mode = mode  # "train" or "test"

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1]), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1]), interpolation=Image.BICUBIC), 
            transforms.Lambda(lambda img: img.convert('I;16')),
            transforms.Lambda(lambda img: torch.tensor(np.array(img, dtype=np.float32)) / 200.0 / 300.0)
                ])

    def __len__(self):
        return len(self.data_df)

    def _flip_intrinsics(self, intr, width):
        # Horizontal flip: cx' = (W - 1) - cx ; fx, fy, cy unchanged
        fx, fy, cx, cy = intr
        cx_flipped = (width - 1) - cx
        return [fx, fy, cx_flipped, cy]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        flip_mask = self.flip_marks[idx]
        poses_gt = self.poses[idx]
        intrinsics_gt = self.intrinsics[idx].copy()  # [fx, fy, cx, cy]

        # ---- read RGBs
        pth = self.imgs_paths[idx]
        img1, img2 = Image.open(pth[0]), Image.open(pth[1])
        if img1.mode == 'RGBA':  # guard
            img1, img2 = img1.convert('RGB'), img2.convert('RGB')
        w1, h1 = img1.size

        # ---- read depths (meters) as torch HxW
        depth0 = read_depth(self.depth_paths[idx][0])
        depth1 = read_depth(self.depth_paths[idx][1])

        # ---- optional horizontal flip (training-time augmentation)
        if flip_mask and self.mode == "train":
            # flip RGBs
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            # flip depths: HxW -> flip dim=1 (W)
            depth0 = depth0.transpose(Image.FLIP_LEFT_RIGHT)
            depth1 = depth1.transpose(Image.FLIP_LEFT_RIGHT)
            # flip intrinsics
            intrinsics_gt = self._flip_intrinsics(intrinsics_gt, w1)

            # optional pose symmetry (keep your convention)
            r_matrix = np.array(poses_gt[6:]).reshape(3, 3)
            r_euler = rotationMatrixToEulerAngles(r_matrix)
            r_matrix_symm = eulerAnglesToRotationMatrix([r_euler[0], -r_euler[1], -r_euler[2]])
            t_symm = [-poses_gt[0], poses_gt[1], poses_gt[2]]
            r_symm = [poses_gt[3], -poses_gt[4], -poses_gt[5]]
            poses_gt = t_symm + r_symm + r_matrix_symm.flatten().tolist()

        poses_gt = torch.tensor(poses_gt, dtype=torch.float32)
        time_freqs = torch.tensor(self.time_freqs[idx], dtype = torch.float32)
        # ---- crop logic (per mode and dataset)
        if self.mode == 'test':
            if 'KITTI' in pth[0]:
                crop_w, crop_h = 658, 370
                start_x = int((184 + 384) / 2); start_y = 0
            elif 'stereo_front_left' in pth[0]:
                crop_w, crop_h = 2048, 1152
                start_x = 0; start_y = 140
            else:
                crop_w, crop_h = w1, h1
                start_x = 0; start_y = 0
        else:  # train
            k = random.choices([1, random.uniform(self.args.scale[0], self.args.scale[1])], [0.3, 0.7])[0]
            crop_w = int(w1 * k)
            crop_h = int(h1 * k)
            start_x = np.random.randint(0, w1 - crop_w + 1)
            start_y = np.random.randint(0, h1 - crop_h + 1)

        # ---- apply crop to RGB (PIL crop is fine for color)
        img1 = img1.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        img2 = img2.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))

        # ---- apply crop to depth
        depth0 = depth0.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        depth1 = depth1.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))

        # ---- intrinsics after crop then resize to model input
        fx, fy, cx, cy = intrinsics_gt
        cx_crop = cx - start_x
        cy_crop = cy - start_y
        sw = self.img_size[1] / float(crop_w)
        sh = self.img_size[0] / float(crop_h)
        fx_resize = fx * sw
        fy_resize = fy * sh
        cx_resize = cx_crop * sw
        cy_resize = cy_crop * sh
        intrs = torch.tensor([fx_resize, fy_resize, cx_resize, cy_resize], dtype=torch.float32)

        # ---- resize RGBs
        img1 = self.img_transform(img1)  # 3xH'xW'
        img2 = self.img_transform(img2)
        imgs = torch.cat((img1, img2), dim=0)  # 6xH'xW'

        # ---- resize depths with bilinear on float tensors
        depth0 = self.depth_transform(depth0).unsqueeze(0)
        depth1 = self.depth_transform(depth1).unsqueeze(0)

        # ---- intrinsic layer & rays_xy at final resolution
        intrinsicLayer = make_intrinsics_layer(
            self.img_size[1], self.img_size[0], fx_resize, fy_resize, cx_resize, cy_resize)  # expect CxHxW (np)
        intrinsicLayer = torch.from_numpy(intrinsicLayer).float().unsqueeze(0)

        pseudo_lidar_xy = make_pseudo_lidar(
            self.img_size[1], self.img_size[0], fx_resize, fy_resize, cx_resize, cy_resize)  # (2,H',W') np
        pseudo_lidar_xy = torch.from_numpy(pseudo_lidar_xy).float()

        # depth0 is (1,H',W'), broadcast to (2,H',W') for XY meters
        pseudo_lidar_xy = pseudo_lidar_xy * depth0  # (2,H',W')

        # ---- stacks
        depth_3d = torch.cat((depth0, intrinsicLayer, pseudo_lidar_xy), dim=0)   # (1+C+2)xH'xW'

        if self.mode == 'test':
            return (pth[0], imgs, poses_gt, intrs, intrinsicLayer, depth0, depth1, depth_3d, time_freqs)
        else:
            return (imgs, poses_gt, intrs, intrinsicLayer, depth0, depth1, depth_3d, time_freqs)

########################################################################################################
#                                          Inference Dataset                                           #
########################################################################################################

def get_infer_data_info(args, infer_data, data_path, depth_path, data_intrinsics, time_freq=10, enforce = None):

    imgs_paths = []
    intrinsics = []
    depth_map_paths = []
    time_freqs = []

    
    for key in infer_data.keys():
        # print(key)
        for scene in infer_data[key]:

            scene_imgs_path = glob.glob(f'{data_path[key]}/sequences/{scene}/image_2/*.jpg') + glob.glob(f'{data_path[key]}/sequences/{scene}/image_2/*.png')
            scene_imgs_path = sorted(scene_imgs_path) # List: N
            ############# Edit HZ
            org_hz = 0
            if 'NUSC' in scene_imgs_path[:-1][0]:
                org_hz = 12
            elif 'KITTI' in  scene_imgs_path[:-1][0]:
                org_hz = 10
            elif 'Argo' in  scene_imgs_path[:-1][0]:
                org_hz = 20
            else:
                org_hz = 10
            time_step = int(org_hz // time_freq)
            scene_poses = []
            scene_freqs = []
            for i in range(0, len(scene_imgs_path)-time_step, time_step):
                scene_poses.append(0) #Degree
                if enforce == None:
                    scene_freqs.append(time_freq)
                else:
                    scene_freqs.append(enforce)

            
            scene_imgs_path_paired = [[scene_imgs_path[i], scene_imgs_path[i+time_step]] for i in range(0, len(scene_imgs_path)-time_step, time_step)] # List: N-1
            # assert len(scene_imgs_path_paired) == len(scene_poses)

            scene_depth_path = glob.glob(f'{depth_path[key]}/sequences/{scene}/image_2/*.png')
            scene_depth_path = sorted(scene_depth_path) # List: N
            scene_depth_path_paired = [[scene_depth_path[i], scene_depth_path[i+time_step]] for i in range(0, len(scene_depth_path)-time_step, time_step)] # List: N-1
            assert len(scene_depth_path_paired) == len(scene_imgs_path_paired)

            _intrinsics = [data_intrinsics[key][scene]] * len(scene_imgs_path_paired)


            imgs_paths = imgs_paths + scene_imgs_path_paired
            intrinsics = intrinsics + _intrinsics
            depth_map_paths = depth_map_paths + scene_depth_path_paired
            time_freqs = time_freqs + scene_freqs

    data = {'imgs_paths': imgs_paths, 'intrinsics': intrinsics, 'depth_paths': depth_map_paths, 'time_freqs':time_freqs, 'scene_poses': scene_poses }
    df = pd.DataFrame(data, columns = ['imgs_paths', 'intrinsics', 'depth_paths', 'time_freqs', 'scene_poses'])

    return df


class InferDataset(Dataset):
    def __init__(self, data_df, img_size):
        self.data_df = data_df
        self.imgs_paths = np.asarray(data_df.imgs_paths)
        self.img_size = img_size  # (H, W)
        self.intrinsics = np.asarray(data_df.intrinsics)
        self.depth_paths = np.asarray(data_df.depth_paths)
        self.time_freqs = np.asarray(data_df.time_freqs)
        self.scene_poses = data_df.scene_poses
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1]), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1]), interpolation=Image.BICUBIC), 
            transforms.Lambda(lambda img: img.convert('I;16')),
            transforms.Lambda(lambda img: torch.tensor(np.array(img, dtype=np.float32)) / 200.0 / 300.0)
                ])        

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        intrinsics_gt = self.intrinsics[idx].copy()  # [fx, fy, cx, cy]
        time_freqs = self.time_freqs[idx]
        time_freqs = torch.tensor(time_freqs, dtype=torch.float32)
        # ---- read RGBs
        pth = self.imgs_paths[idx]
        img1, img2 = Image.open(pth[0]), Image.open(pth[1])
        if img1.mode == 'RGBA':  # guard
            img1, img2 = img1.convert('RGB'), img2.convert('RGB')
        w1, h1 = img1.size

        # ---- read depths (meters) as torch HxW
        depth0 = read_depth(self.depth_paths[idx][0])
        depth1 = read_depth(self.depth_paths[idx][1])

        # ---- crop logic (per mode and dataset)
        if 'KITTI' in pth[0]:
            crop_w, crop_h = 658, 370
            start_x = int((184 + 384) / 2); start_y = 0
        elif 'stereo_front_left' in pth[0]:
            crop_w, crop_h = 2048, 1152
            start_x = 0; start_y = 140
        else:
            crop_w, crop_h = w1, h1
            start_x = 0; start_y = 0
        
        # ---- apply crop to RGB (PIL crop is fine for color)
        img1 = img1.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        img2 = img2.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))

        # ---- apply crop to depth
        depth0 = depth0.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        depth1 = depth1.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))

        # ---- intrinsics after crop then resize to model input
        fx, fy, cx, cy = intrinsics_gt
        cx_crop = cx - start_x
        cy_crop = cy - start_y
        sw = self.img_size[1] / float(crop_w)
        sh = self.img_size[0] / float(crop_h)
        fx_resize = fx * sw
        fy_resize = fy * sh
        cx_resize = cx_crop * sw
        cy_resize = cy_crop * sh
        intrs = torch.tensor([fx_resize, fy_resize, cx_resize, cy_resize], dtype=torch.float32)

        # ---- resize RGBs
        img1 = self.img_transform(img1)  # 3xH'xW'
        img2 = self.img_transform(img2)
        imgs = torch.cat((img1, img2), dim=0)  # 6xH'xW'

        # ---- resize depths with bilinear on float tensors
        depth0 = self.depth_transform(depth0).unsqueeze(0)
        depth1 = self.depth_transform(depth1).unsqueeze(0)

        # ---- intrinsic layer & rays_xy at final resolution
        intrinsicLayer = make_intrinsics_layer(
            self.img_size[1], self.img_size[0], fx_resize, fy_resize, cx_resize, cy_resize)  # expect CxHxW (np)
        intrinsicLayer = torch.from_numpy(intrinsicLayer).float().unsqueeze(0)

        pseudo_lidar_xy = make_pseudo_lidar(
            self.img_size[1], self.img_size[0], fx_resize, fy_resize, cx_resize, cy_resize)  # (2,H',W') np
        pseudo_lidar_xy = torch.from_numpy(pseudo_lidar_xy).float()

        # depth0 is (1,H',W'), broadcast to (2,H',W') for XY meters
        pseudo_lidar_xy = pseudo_lidar_xy * depth0  # (2,H',W')

        # ---- stacks
        depth_3d = torch.cat((depth0, intrinsicLayer, pseudo_lidar_xy), dim=0)   # (1+C+2)xH'xW'
        self.scene_poses = torch.from_numpy(np.stack(self.scene_poses))
        return (pth[0], imgs, intrs, intrinsicLayer, depth0, depth1, depth_3d, time_freqs, self.scene_poses)

