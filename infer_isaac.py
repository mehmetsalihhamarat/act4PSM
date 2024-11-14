# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using robomimic for Isaac Lab environments.")

parser.add_argument('--eval', action='store_true')
parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

# for ACT
parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
parser.add_argument('--temporal_agg', action='store_true')

# Legacy
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-R1-Multi-Fruit-IK-Abs-Direct-v0",
    help="Name of the task.",
)
parser.add_argument("--dataset_dir", nargs="+", help="dataset_dir", required=False)
parser.add_argument("--result_dir", default="/home/user/zhr_workspace/isacc_lab_galaxea/results/random_pos_continus", help="result_dir", required=False)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import torch
import pickle
import numpy as np
import gymnasium as gym

import imageio
from einops import rearrange

# from tools.parser import get_parser
# from scipy.spatial.transform import Rotation as R
# from geometry_msgs.msg import Transform
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

import torch
import numpy as np
import os
import pickle
import argparse
# import matplotlib.pyplot as plt
# from copy import deepcopy
# from tqdm import tqdm
from einops import rearrange

# from utils import load_data # data functions
# from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy
# from constants import TASK_CONFIGS

import IPython
e = IPython.embed


def save_img(input_image, output_path):
    image = (input_image * 255).to("cpu").numpy().astype(np.uint8)
    image = rearrange(image, "c h w -> h w c")
    imageio.imwrite(output_path, image)


class ACTEvaluator(object):
    def __init__(self, args_dict: dict) -> None:
        print(args_dict)
        np.random.seed(args_dict["seed"])

        self.tick_times = 0
        self.ckpt_dir = args_dict["ckpt_dir"]
        self.temporal_agg = args_dict["temporal_agg"]
        self.query_freq = args_dict["chunk_size"]
        self.chunk_size = args_dict["chunk_size"]
        self.camera_names = ["front_rgb", "left_rgb", "right_rgb"]
        self.all_time_actions = torch.zeros(
            [self.chunk_size, self.chunk_size, self.action_dim]
        ).cuda()
        if self.temporal_agg:
            self.query_freq = 1

        # load norm stats
        self._make_norm_stats_processor()

        # load policy
        self.policy = self._make_policy(args_dict)

    def _make_norm_stats_processor(self):
        stats_path = os.path.join(self.ckpt_dir, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        self.pre_process = lambda x: (x - stats["qpos_mean"]) / stats["qpos_std"]
        self.post_process = lambda x: x * stats["action_std"] + stats["action_mean"]

    def _make_policy(self, args_dict: dict):
        # args_dict["use_one_hot_task"] = False  # todo: 暂时默认不开启multi-task

        camera_names = ['rgb_head', 'rgb_left_hand', 'rgb_right_hand']
        # fixed parameters
        # state_dim = 7
        lr_backbone = 1e-5
        backbone = 'resnet18'
        enc_layers = 4 
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args_dict['lr'],
                        'num_queries': args_dict['chunk_size'],
                        'kl_weight': args_dict['kl_weight'],
                        'hidden_dim': args_dict['hidden_dim'],
                        'dim_feedforward': args_dict['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }

        ckpt_path = os.path.join(self.ckpt_dir, "policy_model.pth")
        print("**************ckpt_path: ", ckpt_path)
        policy = ACTPolicy(policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)

        policy.cuda()
        policy.eval()
        print(f"loaded: {ckpt_path}")

        return policy

    def process_obs(self, obs_dict, tf_type):
        qpos_numpy = obs_dict["qpos"]
        qpos = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        curr_images = []
        for k in self.camera_names:
            image = rearrange(obs_dict[k], "h w c -> c h w")
            curr_images.append(image)

        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        return qpos, curr_image



def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    act = ACTEvaluator(vars(args_cli))

    # reset environment
    obs, _ = env.reset()

    episode_idx = 0
    # simulate environment
    while simulation_app.is_running():
        rel_time_idx = act.tick_times % act.chunk_size
        # run everything in inference mode
        count = 0
        with torch.inference_mode():
            init_pos = env.unwrapped.init_pos
            task_id = env.unwrapped.object_id
            if act.tick_times % act.query_freq == 0:
                obs = obs["policy"]
                obs_dict = {
                    k: v.squeeze(0).cpu().numpy()
                    for k, v in obs.items()
                    # if "joint_pos" in k or "rgb" in k
                }
                if obs_dict is None:
                    raise ValueError("Observations are None.")
                # print(f"obs_dict: {obs_dict.keys()}")

                qpos, curr_image = act.process_obs(obs_dict, act.tf_type)

                act.latest_action_buff = act.policy(qpos, curr_image)
                # print(f"latest_action_buff: {act.latest_action_buff}, shape: {act.latest_action_buff.shape}")
                act.all_time_actions[rel_time_idx] = act.latest_action_buff

            if act.latest_action_buff is None:
                raise ValueError("Latest action buffer is None.")

            if act.temporal_agg:
                past_num = min(act.chunk_size, act.tick_times)
                row_idxs = torch.arange(rel_time_idx, rel_time_idx - past_num, -1)
                col_idxs = torch.arange(0, past_num, 1)
                action_for_curr_step = act.all_time_actions[row_idxs, col_idxs]
                action_populated = torch.all(action_for_curr_step != 0, axis=1)
                actions_for_curr_step = action_for_curr_step[action_populated]
                # print(f"actions_for_curr_step: {actions_for_curr_step}")

                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = np.flip(exp_weights / exp_weights.sum()).copy()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(
                    dim=0, keepdim=True
                )
            else:
                raw_action = act.latest_action_buff[:, rel_time_idx]

            # print(f"raw_action: {raw_action}")
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = act.post_process(raw_action)

            action = (
                torch.from_numpy(action)
                .to(device=env.unwrapped.device)
                .view(1, 14)
                .float()
            )
            # print(f"tick times {act.tick_times}, output action {action}")

            # apply actions
            obs, reward, terminated, truncated, info = env.step(action)
            dones = terminated | truncated
            reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
            if dones.any():
                act.all_time_actions = torch.zeros([act.chunk_size, act.chunk_size, act.action_dim]).cuda()
                act.tick_times = 0

                np_saver_success = os.path.join(f"{args_cli.result_dir}", "task_"+str(task_id), "npy", "success")
                np_saver_full = os.path.join(f"{args_cli.result_dir}", "task_"+str(task_id), "npy", "full")
                if not os.path.exists(np_saver_success):
                    os.makedirs(np_saver_success)
                if not os.path.exists(np_saver_full):
                    os.makedirs(np_saver_full)
                if terminated.any():
                    np.save(os.path.join(np_saver_success, f"episode_{episode_idx}.npy"), init_pos.cpu().numpy())
                np.save(os.path.join(np_saver_full, f"episode_{episode_idx}.npy"), init_pos.cpu().numpy())
                

                episode_idx += 1


        act.tick_times += 1

    # close the simulator
    env.close()


if __name__ == "__main__":

    
    # run the main function
    main()
    # close sim app
    simulation_app.close()
