import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyviz3d.visualizer as viz

def plot_route(ax, x_gt, z_gt, x_est, z_est, method, sequence, color, description):
    ax.scatter(x_gt[0], z_gt[0], label='Sequence start', marker='s', color='k')
    ax.plot(x_gt, z_gt, 'k', label='Ground truth', linewidth=2.5)
    ax.plot(x_est, z_est, color + '' + description, label=method, linewidth=3.5)
    ax.legend(loc='upper left', fontsize='x-large')
    ax.grid(visible=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.set_title('Visual Odometry - Sequence ' + sequence, fontdict={'fontsize': 20})
    ax.set_xlabel('X[m]', fontdict={'fontsize': 16})
    ax.set_ylabel('Z[m]', fontdict={'fontsize': 16})
    ax.axis('equal')

def show_pcd(img, lidar_xy, depth, save_path):
    # img:  (3,H_img,W_img) or (B,3,H_img,W_img)
    # depth:(1,Hd,Wd) or (Hd,Wd)   -- metric depth
    # lidar_xy: (2,Hd,Wd) = ((u-cx)/fx, (v-cy)/fy)

    if img.dim()==4: img = img[0]
    if depth.dim()==3: depth = depth[0]

    H_d, W_d = depth.shape
    H_i, W_i = img.shape[-2:]

    if W_i == 2*W_d:        # left|right
        img   = img[..., :W_d]           # keep left (or :W_d for left, W_d: for right)
    elif H_i == 2*H_d:      # top|bottom  
        img   = img[..., :H_d, :]        # keep TOP; use H_d: for bottom

    # ensure all tensors match
    assert img.shape[-2:] == (H_d, W_d)

    Z = depth
    X = lidar_xy[0]
    Y = lidar_xy[1]

    pts = torch.stack([X, Y, Z], -1).reshape(-1,3).cpu().numpy()
    mask = np.isfinite(pts).all(1) & (pts[:,2] > 0)

    colors = img.permute(1,2,0).reshape(-1,3).cpu().numpy()
    if colors.max() <= 1: colors *= 255
    colors = colors.astype(np.float32)[mask]

    vis = viz.Visualizer()
    vis.add_points("pcl", pts[mask].astype(np.float32), colors, point_size=1)
    vis.save(save_path)
    


def visualizer(inference_data, save_path, pth):

    for map, value in inference_data.items():
        for key, _ in value.items():

            poses_dir = '{}/{}/{}/{}'.format(save_path, key, pth.split('/')[-2], pth.split('/')[-1].replace('.pt', ''))
            filenames = [f for f in os.listdir(poses_dir) if f.endswith('.txt')]
            poses_dirs = [os.path.join(poses_dir, f) for f in filenames]
            
            for dir in poses_dirs:
                scene_num = dir.split('/')[-1].split('.')[0]

                if 'KITTI' in map:
                    gt_pose_path = './odom-eval/dataset/kitti/gt_poses/' + dir.split('/')[-1]
                elif 'NUSC_X' in map:
                    gt_pose_path = './odom-eval/dataset/nusc/gt_poses/' + dir.split('/')[-1]
                elif 'ARGO2_Stereo' in map:
                    gt_pose_path = './odom-eval/dataset/argo2_stereo/gt_poses/' + dir.split('/')[-1]
                elif 'GTA' in map:
                    gt_pose_path = './odom-eval/dataset/gta/gt_poses/' + dir.split('/')[-1]
                elif 'NUSC_Mini' in map:
                    gt_pose_path =  '/fs/nexus-projects/AD_dashrecon/ADVO/data/nuScenes/NUSCv1.0-mini_12hz/CAM_FRONT/poses/' + dir.split('/')[-1]
                    
                with open(dir) as f:
                    raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                with open(gt_pose_path) as f:
                    gt_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                        
                x_gt = gt_poses[:, 3]
                z_gt = gt_poses[:, -1]

                x_deepvo = raw_poses[:, 3]
                z_deepvo = raw_poses[:, -1]


                # (x-z plane)

                colors = ['b', 'y', 'r']
                descriptions = ['--', '-.', ':']

                fig, ax = plt.subplots(1, figsize=(12, 12))
                plot_route(ax,
                        x_gt, z_gt,
                        x_deepvo, z_deepvo,
                        'DeepVO',
                        scene_num,
                        colors[0],
                        descriptions[0])

                plt.savefig('{}/{}/{}/{}/{}.png'.format(save_path, key, pth.split('/')[-2], pth.split('/')[-1].replace('.pt', ''), scene_num))
                plt.close(fig)

