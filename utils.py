import numpy as np
import torch
import os
import h5py
import glob
import cv2
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

IMAGE_WIDTH = 128   
IMAGE_HEIGHT = 128
hdf5_File_Name = 'dataset_part1.hdf5'
action_dim = 7

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, img_compressed):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunck_size = chunk_size
        self.img_compressed = img_compressed

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        group_path = f"data/demo_{episode_id}"

        dataset_path = os.path.join(self.dataset_dir, hdf5_File_Name)

        with h5py.File(dataset_path, 'r') as root:
            original_action_shape = root[f'{group_path}/obs/actions'].shape

            episode_len = original_action_shape[0]

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            
            # get observation at start_ts only
            joint_pos = root[f'{group_path}/obs/joint_pos'][start_ts]
            #joint_vel = root[f'{group_path}/obs/joint_vel'][start_ts]
            #joint_effort = root[f'{group_path}/obs/joint_effort'][start_ts]
            qpos = np.concatenate([joint_pos], axis=0)
            
            image_dict = dict()
            for cam_name in self.camera_names:
                if self.img_compressed:
                    # Decode as 3-channel color and convert to RGB
                    origin_image_bytes = root[f'{group_path}/obs/{cam_name}'][start_ts]
                    np_array = np.frombuffer(origin_image_bytes, np.uint8)
                    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        raise RuntimeError(f"Failed to decode compressed image for {group_path}/obs/{cam_name} at {start_ts}")
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_dict[cam_name] = cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
                else: 
                    # Raw array from HDF5 (typically RGB); ensure 3 channels and convert to RGB if needed
                    image = root[f'{group_path}/obs/{cam_name}'][start_ts]
                    # If RGBA, drop alpha
                    if image.ndim == 3 and image.shape[-1] == 4:
                        image = image[..., :3]
                    image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    image_dict[cam_name] = image_resized

            # get all actions after and including start_ts
            is_sim = True
            if is_sim:
                actions = root[f'{group_path}/obs/actions'][start_ts:]
                action_len = episode_len - start_ts
            else:
                actions = root[f'{group_path}/obs/actions'][max(0, start_ts - 1) + 1:]  # hack, to make timesteps more aligned
                action_len = episode_len - (max(0, start_ts - 1) + 1)  # hack, to make timesteps more aligned
            action = np.concatenate([actions], axis=1)

        padded_action = np.zeros((self.chunck_size, original_action_shape[1]), dtype=np.float32)
        if action_len <= self.chunck_size:
            padded_action[:action_len] = action
        else:
            padded_action[:] = action[:self.chunck_size]
        is_pad = np.zeros(self.chunck_size)
        if action_len < self.chunck_size:
            is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # Construct observations
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()            
        action_data = torch.from_numpy(padded_action).float()  
        is_pad = torch.from_numpy(is_pad).bool()              

        # Change image data from channel-last to channel-first for PyTorch [K,H,W,C] -> [K,C,H,W]
        image_data = torch.einsum('k h w c -> k c h w', image_data) 

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, hdf5_File_Name)
        with h5py.File(dataset_path, 'r') as root:
            group_path = f"data/demo_{episode_idx}"
            # Collect joint states
            joint_pos = root[f'{group_path}/obs/joint_pos'][()]
            #joint_vel = root[f'{group_path}/obs/joint_vel'][()]
            #joint_effort = root[f'{group_path}/obs/joint_effort'][()]
            qpos = np.concatenate([joint_pos], axis=1)

            # Collect actions
            action = root[f'{group_path}/obs/actions'][()]
            action = np.concatenate([action], axis=1)
            #action = action[1:] # remove first action (optional)

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)  
    all_action_data = torch.cat(all_action_data, dim=0)
    all_action_data = all_action_data

    # Normalize actions
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-3, np.inf) # clipping

    # Normalize observations
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-3, np.inf) # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(), 
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(), 
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos
    }
    
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size, img_compressed):
    print(f'\nLoading PSM Data from: {dataset_dir}\n')
    # Obtain train-test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    print(f"Total episodes: {num_episodes}")
    print(f"Train episodes: {len(train_indices)}")
    print(f"Validation episodes: {len(val_indices)}")

    # Obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # Construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size, img_compressed)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size, img_compressed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    #print(f"Train batches: {len(train_dataloader)}")
    #print(f"Validation batches: {len(val_dataloader)}")

    return train_dataloader, val_dataloader, norm_stats

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
