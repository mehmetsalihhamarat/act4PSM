import numpy as np
import torch
import os
import h5py
import glob
import cv2
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

IMAGE_WIDTH = 84   
IMAGE_HEIGHT = 84

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

        # Use group path directly from dataset.hdf5
        episode_id = self.episode_ids[index]
        group_path = f"data/demo_{episode_id}"

        dataset_path = os.path.join(self.dataset_dir, 'dataset_2.hdf5')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"dataset.hdf5 not found in directory: {self.dataset_dir}")

        with h5py.File(dataset_path, 'r') as root:
            if group_path not in root:
                raise KeyError(f"Group {group_path} not found in dataset.hdf5")

            length = root[f'{group_path}/obs/joint_pos'].shape[0]
            original_action_shape = (length, 7)  # Your actions shape: (88, 7)
            episode_len = length

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
                
            # get observation at start_ts only
            qpos = root[f'{group_path}/obs/joint_pos'][start_ts]  # Shape: (7,)
            #qpos = root[f'{group_path}/obs/joint_pos'][start_ts, :7]

            #qpos = np.concatenate([qpos], axis=0)  # Shape: (8,)

            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name not in ["rgb_endo_cam", "rgb_wrist_cam"]:
                    raise ValueError(f"Unknown camera name: {cam_name}")

                if self.img_compressed:
                    raise ValueError("Compressed images are not supported in your dataset.")

                # Directly access raw images from your dataset
                image = root[f'{group_path}/obs/{cam_name}'][start_ts]  # Shape: (480, 320, 3)
                image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                image_dict[cam_name] = image_resized

            # get all actions after and including start_ts
            action = root[f'{group_path}/obs/actions'][max(0, start_ts - 1) + 1:]  # Shape: (remaining_steps, 7)
            action_len = episode_len - (max(0, start_ts - 1) + 1)  # Number of remaining timesteps

        padded_action = np.zeros((self.chunck_size, 7), dtype=np.float32)  # (chunk_size, 7)
        if action_len <= self.chunck_size:
            padded_action[:action_len] = action
        else:
            padded_action[:] = action[:self.chunck_size]
        is_pad = np.zeros(self.chunck_size, dtype=np.bool_)
        if action_len < self.chunck_size:
            is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            if cam_name not in image_dict:
                raise KeyError(f"Camera {cam_name} not found in image_dict")
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)  # Shape: (2, 240, 320, 3)

        # Construct observations
        image_data = torch.from_numpy(all_cam_images).float()  # Shape: (2, 240, 320, 3)
        qpos_data = torch.from_numpy(qpos).float()            # Shape: (16,) -> 8 joints
        action_data = torch.from_numpy(padded_action).float()  # Shape: (chunk_size, 7)
        is_pad = torch.from_numpy(is_pad).bool()              # Shape: (chunk_size,)

        # Change image data from channel-last to channel-first for PyTorch
        image_data = torch.einsum('k h w c -> k c h w', image_data)  # Shape: (2, 3, 240, 320)

        # Normalize image values to [0, 1]
        image_data = image_data / 255.0

        # Normalize actions using PSM dataset statistics
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # Normalize joint states (qpos)
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []

    dataset_path = os.path.join(dataset_dir, 'dataset_2.hdf5')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset.hdf5 not found in directory: {dataset_dir}")

    with h5py.File(dataset_path, 'r') as root:
        for episode_idx in range(1, num_episodes + 1):
            group_path = f"data/demo_{episode_idx}"
            if group_path not in root:
                raise KeyError(f"Group {group_path} not found in dataset.hdf5")

            # Collect joint positions and velocities
            #qpos = root[f'{group_path}/obs/joint_pos'][:, :7]  # Shape: (88, 7)
            qpos = root[f'{group_path}/obs/joint_pos'][:]
            qpos_combined = np.concatenate([qpos], axis=1)  # Shape: (88, 7)

            # Collect actions
            actions = root[f'{group_path}/obs/actions'][:]  # Shape: (88, 7)

            all_qpos_data.append(torch.from_numpy(qpos_combined))
            all_action_data.append(torch.from_numpy(actions))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)  # Shape: (total_steps, 7)
    all_action_data = torch.cat(all_action_data, dim=0)  # Shape: (total_steps, 7)

    # Normalize actions
    action_mean = all_action_data.mean(dim=0)
    action_std = all_action_data.std(dim=0).clamp(min=1e-2)

    # Normalize qpos
    qpos_mean = all_qpos_data.mean(dim=0)
    qpos_std = all_qpos_data.std(dim=0).clamp(min=1e-2)

    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": all_qpos_data[0].numpy()
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size, img_compressed):
    print(f'\nLoading PSM Data from: {dataset_dir}\n')

    # Ensure camera names are correct
    for cam_name in camera_names:
        if cam_name not in ["rgb_endo_cam", "rgb_wrist_cam"]:
            raise ValueError(f"Invalid camera name: {cam_name}. Use only 'rgb_endo_cam' or 'rgb_wrist_cam'.")

    # Create a list of episode IDs matching 'data/demo_{id}'
    episode_ids = list(range(1, num_episodes + 1))

    # Obtain train-test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(episode_ids)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    print(f"Total episodes: {num_episodes}")
    print(f"Train episodes: {len(train_indices)}")
    print(f"Validation episodes: {len(val_indices)}")

    # Obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # Construct dataset and dataloader
    train_dataset = EpisodicDataset(
        episode_ids=train_indices,
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        chunk_size=chunk_size,
        img_compressed=img_compressed
    )
    val_dataset = EpisodicDataset(
        episode_ids=val_indices,
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        chunk_size=chunk_size,
        img_compressed=img_compressed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2
    )

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")

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
