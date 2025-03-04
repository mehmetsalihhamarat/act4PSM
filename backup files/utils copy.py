import numpy as np
import torch
import os
import h5py
import glob
import cv2
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

IMAGE_WIDTH = 320   
IMAGE_HEIGHT = 240

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
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        episode_id = f"{episode_id+1:04d}"

        # Define the pattern with a wildcard for the timestamp part
        pattern = os.path.join(self.dataset_dir, f'*-{episode_id}-*.hdf5')

        # Use glob to find the file(s) matching the pattern
        matching_files = glob.glob(pattern)

        # Check if at least one file matches the pattern
        if not matching_files:
            raise FileNotFoundError(f"No file found for pattern: {pattern}")

        # Use the first matching file (or modify to handle multiple matches)
        dataset_path = matching_files[0]

        

        with h5py.File(dataset_path, 'r') as root:
            length = root['/upper_body_observations/left_arm_joint_position'].shape[0]
            original_action_shape = (length, 14)
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos_arm_l = root['/upper_body_observations/left_arm_joint_position'][start_ts]
            qpos_gripper_l = root['/upper_body_observations/left_arm_gripper_position'][start_ts]
            qpos_arm_r = root['/upper_body_observations/right_arm_joint_position'][start_ts]
            qpos_gripper_r = root['/upper_body_observations/right_arm_gripper_position'][start_ts]
            qpos = np.concatenate([qpos_arm_l, qpos_gripper_l, qpos_arm_r, qpos_gripper_r], axis=0)

            image_dict = dict()
            for cam_name in self.camera_names:
                if self.img_compressed:
                    origin_image_bytes = root[f'/upper_body_observations/{cam_name}'][start_ts]
                    np_array = np.frombuffer(origin_image_bytes, np.uint8)
                    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
                    image_dict[cam_name] = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                else:
                    image = root[f'/upper_body_observations/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # get all actions after and including start_ts
            action_arm_l = root['/upper_body_observations/left_arm_joint_position'][max(0, start_ts - 1) + 1:] # hack, to make timesteps more aligned
            action_gripper_l = root['/upper_body_observations/left_arm_gripper_position'][max(0, start_ts - 1) + 1:]
            action_arm_r = root['/upper_body_observations/right_arm_joint_position'][max(0, start_ts - 1) + 1:]
            action_gripper_r = root['/upper_body_observations/right_arm_gripper_position'][max(0, start_ts - 1) + 1:]
            action = np.concatenate([action_arm_l, action_gripper_l, action_arm_r, action_gripper_r], axis=1)
            action_len = episode_len - (max(0, start_ts - 1) + 1) # hack, to make timesteps more aligned

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

        # construct upper_body_observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
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
        episode_idx = f"{episode_idx+1:04d}"
        # Construct the pattern to match files with a wildcard for the timestamp
        pattern = os.path.join(dataset_dir, f'*-{episode_idx}-*.h5')

        # Find files matching the pattern
        matching_files = glob.glob(pattern)

        # Ensure at least one matching file exists
        if not matching_files:
            raise FileNotFoundError(f"No file found for pattern: {pattern}")

        # Use the first matching file
        dataset_path = matching_files[0]

        # Open the file with h5py and process it
        with h5py.File(dataset_path, 'r') as root:
            qpos_arm_l = root['/upper_body_observations/left_arm_joint_position'][()]
            qpos_gripper_l = root['/upper_body_observations/left_arm_gripper_position'][()]
            qpos_arm_r = root['/upper_body_observations/right_arm_joint_position'][()]
            qpos_gripper_r = root['/upper_body_observations/right_arm_gripper_position'][()]
            qpos = np.concatenate([qpos_arm_l, qpos_gripper_l, qpos_arm_r, qpos_gripper_r], axis=1)
            
            action_arm_l = root['/upper_body_observations/left_arm_joint_position'][()]
            action_gripper_l = root['/upper_body_observations/left_arm_gripper_position'][()]
            action_arm_r = root['/upper_body_observations/right_arm_joint_position'][()]
            action_gripper_r = root['/upper_body_observations/right_arm_gripper_position'][()]
            action = np.concatenate([action_arm_l, action_gripper_l, action_arm_r, action_gripper_r], axis=1)
            action = action[1:] # remove first action

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.concatenate(all_qpos_data, dim=0)
    all_action_data = torch.concatenate(all_action_data, dim=0)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size, img_compressed):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size, img_compressed)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size, img_compressed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

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
