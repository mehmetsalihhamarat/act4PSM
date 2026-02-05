# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""
#from icecream import ic
import argparse
import time
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play policy trained using robomimic for Isaac Lab environments.",
    conflict_handler="resolve")

parser.add_argument('--eval', action='store_true')
parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
parser.add_argument('--task_name', action='store', type=str, default="Isaac-Lift-Needle-PSM-IK-Rel-v0", help='task_name', required=True)
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
parser.add_argument('--num_queries', default=100, type=int, # will be overridden
                        help="Number of query slots")

parser.add_argument("--dataset_dir", nargs="+", help="dataset_dir", required=False)
parser.add_argument("--result_dir", default="/home/nural/IsaacLab/results_act/lift_needle", help="result_dir", required=False)

# Legacy
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

print("Arguments already in parser:")
print(parser._option_string_actions.keys())

# parse the arguments
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", True)
setattr(args_cli, "headless", False)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

start_time = time.time()

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
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import argparse
# import matplotlib.pyplot as plt
# from copy import deepcopy
# from tqdm import tqdm

# from utils import load_data # data functions
# from utils import compute_dict_mean, set_seed, detach_dict # helper functions 
from policy import ACTPolicy

# from constants import TASK_CONFIGS

import IPython
e = IPython.embed


import omni.ui as ui
import omni
from isaaclab.devices import Se3Keyboard, Se3SpaceMouse
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

# Timeline interface and global teleop flag
timeline = omni.timeline.get_timeline_interface()
teleop_mode = False
teleop_device = "keyboard"
# Teleop interface
if teleop_device == "keyboard":
    teleopdev = Se3Keyboard(pos_sensitivity=0.002, rot_sensitivity=0.05)
elif teleop_device == "spacemouse":
    teleopdev = Se3SpaceMouse(pos_sensitivity=0.003, rot_sensitivity=0.005)

# Functions for control mode switching
def set_teleop_mode():
    global teleop_mode
    teleop_mode = True
    print("[TELEOP] Entered teleoperation mode.")
    timeline.pause()

def set_rollout_mode():
    global teleop_mode
    teleop_mode = False
    print("[ROLLOUT] Resumed policy rollout.")
    timeline.play()

teleop_confirmation = None  # global flag for user confirmation
awaiting_user_confirmation = False  # tracks if we’re waiting for user input
# Control panel UI
window = ui.Window("Control Panel", width=250, height=100)
with window.frame:
    with ui.VStack():
        ui.Button("TELEOP Mode", clicked_fn=set_teleop_mode)
        ui.Button("ROLLOUT Mode", clicked_fn=set_rollout_mode)

        def confirm_teleop_true():
            global teleop_confirmation, awaiting_user_confirmation
            if awaiting_user_confirmation:
                teleop_confirmation = True
                awaiting_user_confirmation = False
                print("[CONFIRM] Episode marked as TELEOP.")

        def confirm_teleop_false():
            global teleop_confirmation, awaiting_user_confirmation
            if awaiting_user_confirmation:
                teleop_confirmation = False
                awaiting_user_confirmation = False
                print("[CONFIRM] Episode NOT marked as TELEOP.")

        ui.Button("Save Episode", clicked_fn=confirm_teleop_true)
        ui.Button("DON'T Save Episode", clicked_fn=confirm_teleop_false)

import json
import h5py
import os

def delete_non_teleop_episodes(hdf5_path, teleop_flags):
    with h5py.File(hdf5_path, "r+") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        
        # Safety check
        if len(demo_keys) > len(teleop_flags):
            print(f"[WARN] There are more demos in HDF5 ({len(demo_keys)}) than teleop flags ({len(teleop_flags)}).")
        
        keep_keys = []
        for i, key in enumerate(demo_keys):
            if i < len(teleop_flags) and teleop_flags[i]:
                keep_keys.append(key)
            else:
                print(f"[DELETE] Removing non-teleop episode: {key}")
                del f["data"][key]
        
        print(f"[INFO] Kept {len(keep_keys)} episodes: {keep_keys}")


def reorder_recovery_hdf5(hdf5_path, starting_idx):
    """
    Renames demo groups in the HDF5 file starting from `starting_idx`.
    Old demos (demo_0, demo_1, ...) become demo_50, demo_51, etc.

    Args:
        hdf5_path (str): Path to the recovery HDF5 file.
        starting_idx (int): Starting index for renaming (e.g., 50).
    """
    temp_path = hdf5_path.replace(".hdf5", "_temp.hdf5")

    with h5py.File(hdf5_path, "r") as src, h5py.File(temp_path, "w") as dst:
        demos = sorted([k for k in src['data'].keys() if k.startswith("demo_")],
                       key=lambda x: int(x.split("_")[1]))

        dst.create_group("data")
        for new_id, old_key in enumerate(demos):
            new_key = f"demo_{starting_idx + new_id}"
            src.copy(f"data/{old_key}", dst["data"], name=new_key)

        # Optionally, copy non-demo metadata if any
        for k in src.keys():
            if k != "data":
                src.copy(k, dst)

    # Replace the original file
    os.replace(temp_path, hdf5_path)
    print(f"Reordering complete. Demos renamed starting from demo_{starting_idx}.")


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
        self.camera_names = ['rgb_endo_cam','rgb_wrist_cam']
        self.action_dim = 7
        self.all_time_actions = torch.zeros(
            [self.chunk_size, self.chunk_size, self.action_dim]
        ).cuda()

        if self.temporal_agg:
            self.query_freq = 15

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

        camera_names = ['rgb_endo_cam','rgb_wrist_cam']
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

        custom_ckpt = "student_hs_last.ckpt"
        read_seed = args_dict["seed"]
        read_ckpt_path = "best"         # any epoch number, latest or best policy
        if read_ckpt_path == "best":
            ckpt_path = os.path.join(self.ckpt_dir, "policy_best.ckpt")
        elif read_ckpt_path == "student":
            ckpt_path = os.path.join(self.ckpt_dir, "student_best.ckpt")
        elif read_ckpt_path == "custom":
            ckpt_path = os.path.join(self.ckpt_dir, custom_ckpt)
        else: 
            ckpt_path = os.path.join(self.ckpt_dir, f"policy_epoch_{read_ckpt_path}_seed_{read_seed}.ckpt")
        #ckpt_path = os.path.join(self.ckpt_dir, "policy_best.ckpt")
        print("**************ckpt_path: ", ckpt_path)
        policy = ACTPolicy(policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        # policy = torch.load(ckpt_path)
        print(loading_status)

        policy.cuda()
        policy.eval()
        print(f"loaded: {ckpt_path}")

        return policy

    def process_obs(self, obs_dict):

        joint_pos_numpy = obs_dict["joint_pos"]
        #joint_vel_numpy = obs_dict["joint_vel"]
        #joint_effort_numpy = obs_dict["joint_effort"]

        qpos_numpy = np.concatenate([joint_pos_numpy], axis=0)

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
    """
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    """
    env_cfg = parse_env_cfg(
        args_cli.task_name,
        device="cuda:0" if not args_cli.cpu else "cpu",  # Use "cuda:0" if not using CPU
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    save_hdf5 = False
    if save_hdf5:
        # Directory and filename for output
        env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()  # type: ignore
        env_cfg.recorders.dataset_export_dir_path = "C:\IsaacLab/recovery_datasets"  # customize as needed
        env_cfg.recorders.dataset_filename = "teleop_recovery.hdf5"
    else:
        print("[SKIP] HDF5 recording is disabled by save_hdf5 flag!")

    # create environment
    env = gym.make(args_cli.task_name, cfg=env_cfg)

    act = ACTEvaluator(vars(args_cli))

    # reset environment
    env.unwrapped.recorder_manager.reset()
    obs, _ = env.reset()
    
    recent_lifting_values = []
    episode_idx = 0
    success_count = 0
    lift_count = 0
    prev_teleop_mode = False 
    latest_obs = None
    teleop_occurred_during_episode = False
    episode_teleop_flags = []
    # simulate environment
    while simulation_app.is_running():

        global teleop_mode

        rel_time_idx = act.tick_times % act.chunk_size
        # run everything in inference mode
        count = 0
        with torch.inference_mode():
            #init_pos = env.unwrapped.init_pos (init_state)
            #init_pos = torch.zeros(8, device=env.unwrapped.device)
            #task_id = env.unwrapped.object_id
            task_id = "psm"
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

                qpos, curr_image = act.process_obs(obs_dict)

                ##act.latest_action_buff = act.policy(qpos, curr_image)   
                
                policy_out = act.policy(qpos, curr_image)
                if isinstance(policy_out, dict):
                    act.latest_action_buff = policy_out['action_pred']
                else:
                    act.latest_action_buff = policy_out
                
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
            last_action = action[-1]

            # Check teleop_mode state transitions and handle appropriately
            if teleop_mode:
                # Store the last observation when switching into teleop mode
                if not prev_teleop_mode:
                    latest_obs = obs  # save the latest obs for smooth transition
                    gripper_command = last_action
                    print("[TELEOP] Transitioning from rollout. Observation state stored.")
                teleop_occurred_during_episode = True

                delta_pose, gripper_command = teleopdev.advance()
                delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                gripper_vel = torch.tensor([[-1 if gripper_command else 1]], dtype=torch.float, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                #gripper_vel = torch.tensor([last_action], dtype=torch.float, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                action = torch.cat([delta_pose, gripper_vel], dim=1)

            else:
                action = (
                    torch.from_numpy(action)
                    .to(device=env.unwrapped.device)
                    .view(1, 7)
                    .float()
                )
            """
            noisy_action = False
            if noisy_action:
                # A bit uniform noise on action
                first_6_actions = action[:6] * np.random.uniform(0.95, 1.05)
                last_action = action[6]
                first_6_actions = torch.from_numpy(first_6_actions).float().to(device=env.unwrapped.device)
                final_action = torch.cat([first_6_actions, torch.tensor([last_action]).float().to(device=env.unwrapped.device)])
                # Apply to IsaacSim
                action = (
                    final_action.view(1, 7)
                )
            else:
                action = (
                    torch.from_numpy(action) # DO NOT FORGET
                    .to(device=env.unwrapped.device)
                    .view(1, 7)
                    .float()
                )
            #print(f"tick times {act.tick_times}, output action {action}")
            """

            # apply actions
            obs, reward, terminated, truncated, info = env.step(action)
            dones = terminated | truncated
            # Automatically switch back to rollout mode if teleop ends with termination
            if teleop_mode and dones.any():
                print("[TELEOP] Episode terminated in teleop mode. Switching back to rollout.")
                set_rollout_mode()
            reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)

            # Access the lifting object reward term directly
            lifting_reward_value = env_cfg.rewards.lifting_object.func(
                env.unwrapped, **env_cfg.rewards.lifting_object.params
            )
            
            # Add the value to the list (keep only the last 10 for safety)
            recent_lifting_values.append(lifting_reward_value)
            if len(recent_lifting_values) > 10:
                recent_lifting_values.pop(0)

            # Success counter
            if dones.any():

                #if teleop_occurred_during_episode:
                    #print("[SAVED] teleop: episode is saved.")
                    #env.unwrapped.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    #env.unwrapped.recorder_manager.export_episodes([0])
                #else:
                    #print("[SKIP] No teleop: episode not saved.")

                
                act.all_time_actions = torch.zeros([act.chunk_size, act.chunk_size, act.action_dim]).cuda()
                act.tick_times = 0
                """
                np_saver_success = os.path.join(f"{args_cli.result_dir}", "task_"+str(task_id), "npy", "success")
                np_saver_full = os.path.join(f"{args_cli.result_dir}", "task_"+str(task_id), "npy", "full")
                if not os.path.exists(np_saver_success):
                    os.makedirs(np_saver_success)
                if not os.path.exists(np_saver_full):
                    os.makedirs(np_saver_full)
                if terminated.any():
                    np.save(os.path.join(np_saver_success, f"episode_{episode_idx}.npy"), init_pos.cpu().numpy())
                np.save(os.path.join(np_saver_full, f"episode_{episode_idx}.npy"), init_pos.cpu().numpy())
                """
                ## Success counter
                if terminated.any():
                    # Instead of saving, print if episode is successful or not
                    print(f'Episode_{episode_idx}: TRUE')
                    success_count += 1
                else:
                    print(f'Episode_{episode_idx}: FALSE')
                
                ### LIFT CONTROL
                # Print the 5th value from the last, if it exists
                if len(recent_lifting_values) >= 5:          
                    lift_reward = int(recent_lifting_values[-2].item())
                    if lift_reward > 0:
                        print(f"LIFT: True")
                        lift_count += 1
                    else: 
                        print(f"LIFT: False")
                else:
                    print("Not enough values to show the 5th from the last")
                # Clear the list for the next episode
                recent_lifting_values.clear() 
                ### ENDS       

                
                total_episodes = episode_idx+1
                success_rate = success_count / total_episodes 
                success_rate_lift = lift_count / total_episodes 
                print(f"Success Rate:{np.round(success_rate*100,1)}%")
                print(f"Success Lift:{np.round(success_rate_lift*100,1)}%")
                
                #episode_teleop_flags.append(teleop_occurred_during_episode)
                if teleop_occurred_during_episode and save_hdf5:
                    global awaiting_user_confirmation, teleop_confirmation
                    print("[WAITING] Please confirm if this was a TELEOP episode using the Control Panel.")
                    awaiting_user_confirmation = True
                    teleop_confirmation = None

                    # Wait for user to click a button
                    while awaiting_user_confirmation:
                        simulation_app.update()

                    episode_teleop_flags.append(teleop_confirmation)
                else:
                    episode_teleop_flags.append(False)

                teleop_occurred_during_episode = False
                #print("[DEBUG] FINAL episode_teleop_flags before deletion:", episode_teleop_flags)

                # Close after x episodes
                rollout = 50 # This defines the number of episodes executed
                if (total_episodes == rollout):
                    print('========================================')
                    print(f"Success Rate after {rollout} episodes: {success_rate * 100:.1f}%")
                    print('========================================')
                    print(f"Success Lift after {rollout} episodes: {success_rate_lift * 100:.1f}%")
                    print('========================================')
                    env.close()
                    if save_hdf5:
                        delete_non_teleop_episodes("C:\IsaacLab/recovery_datasets/teleop_recovery.hdf5", episode_teleop_flags)
                        reorder_start_number = 10
                        reorder_recovery_hdf5("C:\IsaacLab/recovery_datasets/teleop_recovery.hdf5", reorder_start_number)
                    simulation_app.close()
                    #break

                episode_idx += 1

        prev_teleop_mode = teleop_mode
        act.tick_times += 1
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.2f}s")

