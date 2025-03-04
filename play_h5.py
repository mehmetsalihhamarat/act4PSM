import os
import glob
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from einops import rearrange
import argparse
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy
from constants import TASK_CONFIGS

import IPython
e = IPython.embed
import h5py
import cv2

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

def visualize_differences_4(qpos_list, gt_list, infer_list, gt2_list=None, plot_path=None, ylim=None, label_overwrite=None, plot_name = None):
    if label_overwrite:
        if len(label_overwrite) == 4:
            label1, label2,label3,label4 = label_overwrite
        elif len(label_overwrite) == 3:
            label1, label2,label3 = label_overwrite
    else:
        #'Ground Truth', 'Inferred'#'State', 'Command'#,'differences'
        label1, label2,label3,label4 = 'State','Ground Truth Host Command','Inferred Command', 'Ground Truth 2 Command'
    qpos=np.array(qpos_list)
    gt = np.array(gt_list) # ts, dim
    if gt2_list is not None:
        gt2 = np.array(gt2_list) # ts, dim
    infer = np.array(infer_list)
    num_ts, num_dim = gt.shape
    
    STATE_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]
    
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):#the real joint angles/states
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()
    # plot arm command
    for dim_idx in range(num_dim):#the action/the desired/the expected joint angles/states
        ax = axs[dim_idx]
        ax.plot(gt[:, dim_idx], label=label2)
        ax.legend()
    # plot arm command
    for dim_idx in range(num_dim):  # the inferred action
        ax = axs[dim_idx]
        ax.plot(infer[:, dim_idx], label=label3)
        ax.legend()
    if gt2_list is not None:
        for dim_idx in range(num_dim):#the action/the desired/the expected joint angles/states NORMALIZED
            ax = axs[dim_idx]
            ax.plot(gt2[:, dim_idx], label=label4)
            ax.legend()
    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    plot_path_name = os.path.join(plot_path,plot_name)
    plt.savefig(plot_path_name)
    print(f'Saved qpos plot to: {plot_path_name}')
    plt.close()


def get_image(image_dict, camera_names, t, img_compressed):
    curr_images = []
    for cam_name in camera_names:
        if img_compressed:
            curr_image_bytes = image_dict[cam_name][t]
            np_array = np.frombuffer(curr_image_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        else:
            image = image_dict[cam_name][t]
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        curr_image = rearrange(image, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc_h5(config, ckpt_name, camera_names, save_episode=True, dataset_dir=None):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    task_name = config['task_name']
    img_compressed = config['img_compressed']
    state_dim = 14
    policy_class = config['policy_class']
    temporal_agg = config['temporal_agg']
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4 
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': config['lr'],
                        'num_queries': config['chunk_size'],
                        'kl_weight': config['kl_weight'],
                        'hidden_dim': config['hidden_dim'],
                        'dim_feedforward': config['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }

    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    query_frequency = config['chunk_size'] 

    query_frequency = 1
    num_queries = config['chunk_size']
   
    files = list()
    files.extend(glob.glob(os.path.join(dataset_dir, task_name, '*.h5'), recursive=True))
    files = sorted(files)
    episode_id = 0
    for filename in files:
        with h5py.File(filename, 'r') as root:
            qpos_arm_l = root['/upper_body_observations/left_arm_joint_position'][()]
            qpos_gripper_l = root['/upper_body_observations/left_arm_gripper_position'][()]
            qpos_arm_r = root['/upper_body_observations/right_arm_joint_position'][()]
            qpos_gripper_r = root['/upper_body_observations/right_arm_gripper_position'][()]
            qpos_gt = np.concatenate([qpos_arm_l, qpos_gripper_l, qpos_arm_r, qpos_gripper_r], axis=1)
            action_gt = qpos_gt[1:] # remove first action
            image_dict = dict()
            for cam_name in camera_names:
                image_dict[cam_name] = root[f'/upper_body_observations/{cam_name}']
            ### evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros([num_queries, num_queries, state_dim]).cuda() 
            target_qpos_list = []
            with torch.inference_mode():
                for t in range(action_gt.shape[0]):
                    ### update onscreen render and wait for DT
                    qpos_numpy = qpos_gt[t]
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    curr_image = get_image(image_dict, camera_names, t, img_compressed)
                    ### query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)
                        if temporal_agg:
                            all_time_actions[t % num_queries] = all_actions  
                            if (t >= num_queries - 1):
                                rowindex = torch.arange(num_queries)
                                columnindex = (torch.arange(t, t - num_queries, -1)) % num_queries
                            else:
                                rowindex = torch.arange(t + 1)
                                columnindex = torch.arange(t, -1, -1)
                            actions_for_curr_step = all_time_actions[rowindex, columnindex]  
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    else:
                        raise NotImplementedError
                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action
                    target_qpos_list.append(target_qpos)
            action_infer = np.array(target_qpos_list)

            print("file name:", filename)
            visualize_differences_4(qpos_gt, action_gt, action_infer, \
                                    plot_path = os.path.join(dataset_dir, "plot", task_name), \
                                    label_overwrite = ['state','action_gt','infer'], \
                                    plot_name = f'episode_{episode_id}_qpos_qf_{query_frequency}.png')
        episode_id += 1

def count_h5_files(dataset_dir):
    # 查找目录下所有的.h5文件
    h5_files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    # 返回文件数量
    return len(h5_files)
        
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def main(args_dict, dataset_dir):
    set_seed(args_dict["seed"])
    # command line parameters

    num_episodes = 0
    for directory in dataset_dir:
        num_episodes += count_h5_files(directory)

    task_name = args_dict['task_name']
    camera_names = TASK_CONFIGS[task_name]['camera_names']
    
    ckpt_name = f'policy_best.ckpt'
    eval_bc_h5(args_dict, ckpt_name, camera_names, save_episode=True, dataset_dir=dataset_dir)


if __name__ == '__main__':
    dataset_dir = 'data'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)

    # for ACT
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--img_compressed', action='store_true')


    main(vars(parser.parse_args()), dataset_dir)
