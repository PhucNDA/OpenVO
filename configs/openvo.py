from data.argo2_logs_map import argo2_logs_map, argo2_stereo_remove, argo2_openvo, argo2_openvo_test
from data.nusc_scene_map import nusc_scene_map
import json
import os

class Parameters():
	def __init__(self):
		##############################################
		'''
		Dataset Path
		'''
		##############################################

		self.path2_prefix = './data'

		self.data_path =  {
			# 'YouTube': self.path2_prefix + '/YTB/10Hz',
			'ARGO2_Stereo': self.path2_prefix + '/Argoverse_2/20hz/stereo_front_left/test',
            'NUSC': self.path2_prefix + '/nuScenes/NUSCv1.0-trainval_12hz/CAM_FRONT',
			'NUSC_X': self.path2_prefix + '/nuScenes/NUSCv1.0-trainval_12hz/CAM_FRONT',
            'KITTI': self.path2_prefix + '/KITTI',
			'KITTI_X': self.path2_prefix + '/KITTI',
            }

		# Depths
		self.depth_path = {
			# 'YouTube': self.path2_prefix + '/YTB/10Hz/depth_est_intrs',
			'ARGO2_Stereo': self.path2_prefix + '/Argoverse_2/20hz/stereo_front_left/test/depth_gt_intrs',
            'NUSC': self.path2_prefix + '/nuScenes/NUSCv1.0-trainval_12hz/CAM_FRONT/depth_gt_intrs',
			'NUSC_X': self.path2_prefix + '/nuScenes/NUSCv1.0-trainval_12hz/CAM_FRONT/depth_est_intrs',
            'KITTI': self.path2_prefix + '/KITTI/depth_est_intrs', # not yet for GT
			'KITTI_X': self.path2_prefix + '/KITTI/depth_est_intrs',
            }
		
		# Intrinsics
		# f = open(self.data_path['YouTube'] + '/ytb_est_intrs.json') 
		# ytb_intrinsics = json.load(f)
		f = open(self.data_path['ARGO2_Stereo'] + '/av2_est_intrs.json') 
		argo2_stereo_intrinsics = json.load(f)
		f = open(self.data_path['NUSC'] + '/nuscene_gt_intrs.json') 
		nusc_gt_intrinsics = json.load(f)
		f = open(self.data_path['NUSC_X'] + '/nuscene_est_intrs.json') 
		nusc_est_intrinsics = json.load(f)	
		f = open(self.data_path['KITTI'] + '/kitti_gt_intrs.json') 
		kitti_gt_intrinsics = json.load(f)
		f = open(self.data_path['KITTI_X'] + '/kitti_est_intrs.json') 
		kitti_est_intrinsics = json.load(f)

		self.data_intrinsics = {
			# 'YouTube': ytb_intrinsics,
			'ARGO2_Stereo': argo2_stereo_intrinsics,
			'NUSC': nusc_gt_intrinsics,
			'NUSC_X': nusc_est_intrinsics,
			'KITTI': kitti_gt_intrinsics,
			'KITTI_X': kitti_est_intrinsics,
            }

		# Training set
		self.training_data = {
			'NUSC': nusc_scene_map['singapore-onenorth'], 		# gtruth (training)
            # 'NUSC_X': nusc_scene_map['singapore-onenorth'], 	# estimated
			# 'YouTube': [str(i).zfill(2) for i in range(49)]
            }
		# Validation set
		self.testing_data = {
			'KITTI': {'KITTI': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']},
		}
		# Testing set ~ Benchmark
		self.inference_data = {
			# 'YouTube':{'YouTube': ['01', '02']}
			# 'ARGO2_Stereo':{'ARGO2_Stereo': argo2_openvo},
			# 'KITTI': {'KITTI': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']}, # gt-intrinsic
			'KITTI_X': {'KITTI_X': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']}, # wild-camera
			# 'NUSC_X':{'NUSC_X': nusc_scene_map['boston-seaport']}, # subset for fast validation (not benchmark)
			# 'NUSC_X':{'NUSC_X': nusc_scene_map['boston-seaport'] + nusc_scene_map['singapore-queenstown'] + nusc_scene_map['singapore-hollandvillage']},
			# 'NUSC':{'NUSC': nusc_scene_map['boston-seaport'] + nusc_scene_map['singapore-queenstown'] + nusc_scene_map['singapore-hollandvillage']},

		}	

		############################################
		'''
		Data Preprocessing
		'''
		##############################################
		self.n_processors = 8 # suggested: 16
		self.scale = [0.7, 1]
		self.img_w = 640  
		self.img_h = 384   

		##############################################
		'''
		Training
		'''
		##############################################
		self.velocity_target = False # predicting velocity $Future feature$
	
		self.model_type = 'openvo'
		self.adaptiveHZ = [12, 6, 4] # static Hz
		self.dist = False # Distributed Training $Future feature$
		self.step_scheduler = False # Scheduler Step?
		self.freq_k = 8 # time freq pos embed

		self.seed = 2023
		self.epochs = 150
		self.learning_rate = 0.001
		self.batch_size = 16
		self.ssl_pretrain = False
		self.load_ckpt = None
		self.data_flip = True
		self.k = 2
		self.feat_patch_size = (32, 32)
		self.img_patch_size = (128, 128)
		self.attn_embed_dim = 768
		self.mlp_embed_dim = 256 # 256 or 512
		self.num_heads = 4
		self.num_blocks = 8 # suggested ~ 4
		self.model_path = './weights/OpenVO/openvo_nusc_gt_test' # (trained on nusc_gt)
		self.pretrained_flownet_path = './weights/init_weights'


args = Parameters()
