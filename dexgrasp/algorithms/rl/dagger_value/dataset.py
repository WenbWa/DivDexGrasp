import os
import time
import tqdm
import cv2 as cv
import numpy as np
from utils.general_utils import *
from torch.utils.data import Dataset


class ObjectTrajectoryDatasetBatch(Dataset):
    """
        Object Trajectory Dataset: Nobj x Ntraj x {(Nstep, Nobs), (Nstep, Nact), (Nstep, 1), (1,)}
        Object Trajectory Dataset: 3200 x  1000 x {(  200,  300), (  200,   24), (  200, 1), (1,)}
    """
    def __init__(self, config, log_dir, asset_dir, trajectory_dir, object_scale_yaml, target_object_lines, dtype=torch.float32, device='cuda:0'):

        # init dataset info
        self.dtype = dtype
        self.device = device
        self.config = config
        self.log_dir = log_dir
        self.asset_dir = asset_dir
        self.trajectory_dir = trajectory_dir
        self.target_object_lines = target_object_lines
        # locate object mesh and feature folder
        self.object_mesh_dir = osp.join(self.asset_dir, 'meshdatav3_scaled')
        self.visual_feat_dir = osp.join(self.asset_dir, 'meshdatav3_pc_feat')
        # load object_scale_yaml
        self.object_line_list, self.object_scale_list, self.object_scale_dict = load_object_scale_result_yaml(object_scale_yaml)
        # self.object_scale_dict = {object_code: [scale], }
        # self.object_scale_list = [object_code/scale, ]
        # self.object_line_list = [object_line, ]

        # init valid object scales
        self.scale2str = {0.06: '006', 0.08: '008', 0.10: '010', 0.12: '012', 0.15: '015'}
        self.str2scale = {'006': 0.06, '008': 0.08, '010': 0.10, '012': 0.12, '015': 0.15}

        # get object number and trajectory number
        self.num_object = len(target_object_lines)
        self.num_trajectory = self.config['Offlines']['num_trajectory'] if 'num_trajectory' in self.config['Offlines'] else 1000

        # get train_epochs and train_batchs
        self.train_epochs = self.config['Offlines']['train_epochs']  # 20 / 50
        self.train_batchs = self.config['Offlines']['train_batchs']  # 100 / 200
        # get train_iterations
        self.train_iterations = self.num_object * self.num_trajectory // self.train_batchs
        # set log_iterations
        self.log_times = 1 if self.num_object == 1 else 10
        self.log_iterations = self.train_iterations // self.log_times

        # set trajectory group number
        self.group_size = 10
        # get sample length: total_trajectory // group_size
        self.sample_length = self.num_object * (self.num_trajectory // self.group_size)

        # set loading hyper params
        self.load_values = False
        # load dynamic object visual features
        self.load_dynamic_visual_feats = True if 'dynamic_object_visual_feature' in self.config['Offlines'] and self.config['Offlines']['dynamic_object_visual_feature'] else False
        # load static object visual feature
        self.load_static_visual_feats = not self.config['Offlines']['zero_object_visual_feature'] and not self.load_dynamic_visual_feats
        # load static_object_visual_feats (Nobj, 64)
        self._load_static_object_visual_feats()

        # encode object ids as features
        self.encode_object_ids = self.config['Offlines']['encode_object_id_feature'] if 'encode_object_id_feature' in self.config['Offlines'] else False
        self.encode_object_hots = self.config['Offlines']['encode_object_id_hotvect'] if 'encode_object_id_hotvect' in self.config['Offlines'] else False
        # generate object_id_feats
        self.object_id_feats = torch.tensor(self.target_object_lines)
        self.object_id_feats = torch.cat([self.object_id_feats.unsqueeze(-1) * 0., compute_time_encoding(self.object_id_feats, 28)], dim=-1)

        # zero observation forces
        self.zero_obs_forces = self.config['Offlines']['zero_obs_forces'] if 'zero_obs_forces' in self.config['Offlines'] else False
        # zero observation actions
        self.zero_obs_actions = self.config['Offlines']['zero_obs_actions'] if 'zero_obs_actions' in self.config['Offlines'] else False
        # zero observation objects
        self.zero_obs_objects = self.config['Offlines']['zero_obs_objects'] if 'zero_obs_objects' in self.config['Offlines'] else False

        # save config yaml
        save_yaml(os.path.join(self.log_dir, 'train.yaml'), self.config)

        # vision_based training: use_external_feature, update observations
        self.vision_based = True if 'vision_based' in self.config['Modes'] and self.config['Modes']['vision_based'] else False

        # replace Obs with Obs_Dataset: load saved observation from dataset
        if 'Obs_Dataset' in self.config: self.config['Obs'] = self.config['Obs_Dataset']
        # use_external_feature, locate external_feature name and size
        self.use_external_feature = True if 'use_external_feature' in self.config['Offlines'] and self.config['Offlines']['use_external_feature'] else False
        self.external_feature_name = self.config['Offlines']['external_feature_name'] if self.use_external_feature else None
        # self.external_feature_size = int(self.external_feature_name.split('_')[-1]) if self.use_external_feature else None


    def __len__(self):
        # return sampled data_size
        return self.sample_length
    
    def __getitem__(self, idx):
        # locate object and trajectory file
        nobj = idx // (self.num_trajectory // self.group_size)
        ntraj = idx % (self.num_trajectory // self.group_size)
        # load object_trajectory data
        sample = self._load_object_trajectory(nobj, ntraj)
        return sample
    
    # load object trajectory data
    def _load_object_trajectory(self, nobj, ntraj):
        # load object_trajectory_data: {'observations': (Ngroup, 200, Nobs), 'actions': (Ngroup, 200, Nact), 'features': (Ngroup, 200, 64), 'values': (Ngroup, 200, 1), 'valids': (Ngroup, 200, 1), 'successes: (Ngroup, 1, )'}
        object_trajectory_data_path = osp.join(self.trajectory_dir, '{:04d}_seed0'.format(self.target_object_lines[nobj]), 'trajectory/trajectory_{:03d}.pkl'.format(ntraj))
        object_trajectory_data = load_pickle(object_trajectory_data_path)
        
        # load dynamic_object_visual_feature
        if self.load_dynamic_visual_feats:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][0]:self.config['Obs']['intervals']['object_visual'][1]] = object_trajectory_data['features']
        # load static_object_visual_feats
        if self.load_static_visual_feats:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][0]:self.config['Obs']['intervals']['object_visual'][1]] = 0.1 * self.static_object_visual_feats[nobj, :]
        
        # mask_object_state
        if 'mask_object_state' in self.config['Offlines'] and self.config['Offlines']['mask_object_state']:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+3:self.config['Obs']['intervals']['objects'][1]] *= 0.
        # mask_object_velocity
        if 'mask_object_velocity' in self.config['Offlines'] and self.config['Offlines']['mask_object_velocity']:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+6:self.config['Obs']['intervals']['objects'][1]] *= 0.
        # mask_hand_object_state
        if 'mask_hand_object_state' in self.config['Offlines'] and self.config['Offlines']['mask_hand_object_state']:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['hand_objects'][0]:self.config['Obs']['intervals']['hand_objects'][1]] *= 0.
        
        # mask_obs_time
        if 'mask_obs_time' in self.config['Offlines'] and self.config['Offlines']['mask_obs_time']:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['times'][0]:self.config['Obs']['intervals']['times'][1]] *= 0.
        # mask_obs_hand_object
        if 'mask_obs_hand_object' in self.config['Offlines'] and self.config['Offlines']['mask_obs_hand_object']:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['hand_objects'][0]:self.config['Obs']['intervals']['hand_objects'][1]] *= 0.
        # mask_obs_object_visual
        if 'mask_obs_object_visual' in self.config['Offlines'] and self.config['Offlines']['mask_obs_object_visual']:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][0]:self.config['Obs']['intervals']['object_visual'][1]] *= 0.
        # mask_obs_object_state
        if 'mask_obs_object_state' in self.config['Offlines'] and self.config['Offlines']['mask_obs_object_state']:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][1]] *= 0.
        
        # vision_based: update observations
        if self.vision_based:
            # load rendered object_state: features, centers, hand_object
            object_state_path = osp.join(self.trajectory_dir, '{:04d}_seed0'.format(self.target_object_lines[nobj]), 'pointcloud/pointcloud_{:03d}.pkl'.format(ntraj))
            object_state = load_pickle(object_state_path)
            # check valid appears within trajectory
            object_state = check_object_valid_appears(object_state['valids'], object_state)
            # update object features
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][0]:self.config['Obs']['intervals']['object_visual'][1]] = object_state['features']
            # update object centers
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][1]] *= 0
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][0]+3] = object_state['centers']
            # update hand_objects
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['hand_objects'][0]:self.config['Obs']['intervals']['hand_objects'][1]] = object_state['hand_object']
            # update valids with appears
            object_trajectory_data['valids'] *= object_state['appears']
            # use object pcas, estimated from rendered object points
            if 'use_object_pcas' in self.config['Offlines'] and self.config['Offlines']['use_object_pcas']:
                # get object pcas
                object_pcas = object_state['pcas'].reshape(object_state['pcas'].shape[0], object_state['pcas'].shape[1], -1)
                # use dynamic or static object pcas
                if self.config['Offlines']['use_dynamic_pcas']: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+6:self.config['Obs']['intervals']['objects'][0]+15] = object_pcas
                else: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+6:self.config['Obs']['intervals']['objects'][0]+15] = object_pcas[:, 0, :][:, None, :]

        
        # load external feature: state-based or vision-based
        if self.use_external_feature:
            # use state-based pc features
            if self.external_feature_name == 'feature_state_64':
                object_features = object_trajectory_data['features']
            # load external feature
            else: 
                feature_data_path = osp.join(self.trajectory_dir, '{:04d}_seed0'.format(self.target_object_lines[nobj]), '{}/feature_{:03d}_norm.pkl'.format(self.external_feature_name, ntraj))
                object_features = load_pickle(feature_data_path)
            
            # vision_based: update observations
            if self.vision_based:
                # load object_state: object_centers, hand_object_dists
                object_state_path = osp.join(self.trajectory_dir, '{:04d}_seed0'.format(self.target_object_lines[nobj]), 'objectstate/objectstate_{:03d}.pkl'.format(ntraj))
                object_state = load_pickle(object_state_path)
                # check valid, appears within trajectory
                object_state['object_features'] = object_features
                object_state = check_object_valid_appears(object_trajectory_data['valids'], object_state)
                # update object_features
                object_features = object_state['object_features']
                # use state-based object_center 
                if 'use_state_object_center' in self.config['Offlines'] and self.config['Offlines']['use_state_object_center']:
                    object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+3:self.config['Obs']['intervals']['objects'][1]] *= 0
                else:
                    # update objects with object_centers
                    object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][1]] *= 0
                    object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][0]+3] = object_state['object_centers']
                    if 'mask_object_goal' in self.config['Offlines'] and self.config['Offlines']['mask_object_goal']: pass
                    else: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+3:self.config['Obs']['intervals']['objects'][0]+6] = np.array([[[0.0, 0.0, 0.9]]]) - object_state['object_centers']
                # use state-based hand_object
                if 'use_state_hand_object' in self.config['Offlines'] and self.config['Offlines']['use_state_hand_object']: pass
                # update hand_objects
                else: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['hand_objects'][0]:self.config['Obs']['intervals']['hand_objects'][1]] = object_state['hand_object_dists']
                # update valids with appears
                object_trajectory_data['valids'] *= object_state['appears']

                # use object_velocity, estimated from object_centers
                if 'use_object_velocities' in self.config['Offlines'] and self.config['Offlines']['use_object_velocities']:
                    # get previous object_center
                    previous_object_center = np.concatenate([object_state['object_centers'][..., 0][..., None], object_state['object_centers'][..., :-1]], axis=-1)
                    # estimate object_velocity
                    object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+3:self.config['Obs']['intervals']['objects'][0]+6] = object_state['object_centers'] - previous_object_center
                # use object_pcas, estimated from object_points
                if 'use_object_pcas' in self.config['Offlines'] and self.config['Offlines']['use_object_pcas']:
                    # get object_pcas
                    object_pcas = object_state['object_pcas'].reshape(object_state['object_pcas'].shape[0], object_state['object_pcas'].shape[1], -1)
                    # use dynamic or static object_pcas
                    if self.config['Offlines']['use_dynamic_pcas']: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+6:self.config['Obs']['intervals']['objects'][0]+15] = object_pcas
                    else: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+6:self.config['Obs']['intervals']['objects'][0]+15] = object_pcas[:, 0, :][:, None, :]
                # zero object state
                if 'zero_object_state' in self.config['Offlines'] and self.config['Offlines']['zero_object_state']:
                    object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][1]] *= 0.

            # concate external feature
            object_trajectory_data['observations'] = np.concatenate([object_trajectory_data['observations'][..., :self.config['Obs']['intervals']['object_visual'][0]],
                                                                     object_features, object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][1]:]], axis=-1)

        
        # # remove unwanted object_trajectory_data
        # if 'features' in object_trajectory_data : object_trajectory_data.pop('features')
        # if not self.load_values and 'values' in object_trajectory_data: object_trajectory_data.pop('values')

        # zero observation actions
        if self.zero_obs_actions:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['actions'][0]:self.config['Obs']['intervals']['actions'][1]] *= 0.
        # zero observation forces
        if self.zero_obs_forces:
            for name, interval in self.config['Obs']['forces'].items():
                if name not in self.config['Obs']['intervals']: continue
                object_trajectory_data['observations'][..., self.config['Obs']['intervals'][name][0]+interval[0]:self.config['Obs']['intervals'][name][0]+interval[1]] *= 0.
        # zero observation objects, keep object_pos and object_hand_dist
        if self.zero_obs_objects:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+3:self.config['Obs']['intervals']['objects'][1]-3] *= 0.
        
        # encode object id feats
        if self.encode_object_ids:
            # append object id feats into observations
            if len(object_trajectory_data['observations'].shape) == 2: object_id_feats = self.object_id_feats[nobj].repeat(object_trajectory_data['observations'].shape[0], 1)
            else: object_id_feats = self.object_id_feats[nobj].repeat(object_trajectory_data['observations'].shape[0], object_trajectory_data['observations'].shape[1], 1)
            object_trajectory_data['observations'] = np.concatenate([object_trajectory_data['observations'], object_id_feats.numpy()], axis=-1)
        # encode object hot vects
        if self.encode_object_hots:
            if len(object_trajectory_data['observations'].shape) == 2: object_hot_vects = np.zeros((object_trajectory_data['observations'].shape[0], self.config['Offlines']['hotvect_size']))
            else: object_hot_vects = np.zeros((object_trajectory_data['observations'].shape[0], object_trajectory_data['observations'].shape[1], self.config['Offlines']['hotvect_size']))
            object_hot_vects[..., nobj % self.config['Offlines']['hotvect_size']] = 1.
            object_trajectory_data['observations'] = np.concatenate([object_trajectory_data['observations'], object_hot_vects.astype(object_trajectory_data['observations'].dtype)], axis=-1)
        
        # # send object_trajectory_data to GPU tensor
        # for key, value in object_trajectory_data.items():
        #     object_trajectory_data[key] = torch.tensor(object_trajectory_data[key], dtype=self.dtype, device=self.device)
        return object_trajectory_data


    # load static_object_visual_feats (Nobj, 64)
    def _load_static_object_visual_feats(self):
        # init target_object_visual_feats
        self.static_object_visual_feats = np.zeros((self.num_object, 64))
        # load visual_features for target_object_lines
        for nline, line in enumerate(self.target_object_lines):
            # locate object_scale visual_feature
            split_temp = self.object_scale_list[line].split('/')
            object_code, scale_str = '{}/{}'.format(split_temp[0], split_temp[1]), self.scale2str[float(split_temp[2])] 
            file_dir = osp.join(self.visual_feat_dir, '{}/pc_feat_{}.npy'.format(object_code, scale_str))
            # load object_scale visual_feature
            with open(file_dir, 'rb') as file: self.static_object_visual_feats[nline, :] = np.load(file)