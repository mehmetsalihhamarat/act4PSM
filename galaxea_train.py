import os
import glob
import torch
import wandb
import numpy as np
import pickle

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from galaxea_act.config.params import ArmType
from galaxea_act.config.parser import get_parser
from galaxea_act.dataset.episodic_dataset import load_data
from galaxea_act.utlis.utlis import compute_dict_mean, set_seed, detach_dict, load_model_from_checkpoint, get_arm_config
from galaxea_act.algos.act_policy import ACTPolicy


def count_h5_files(dataset_dir):
    # 查找目录下所有的.h5文件
    h5_files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    # 返回文件数量
    return len(h5_files)

def main(args_dict):
    set_seed(args_dict["seed"])
    # command line parameters
    ckpt_dir = args_dict['ckpt_dir']
    with_torso = args_dict['with_torso']

    dataset_dir = args_dict['dataset_dir']

    batch_size_train = args_dict['batch_size']
    batch_size_val = args_dict['batch_size']
    num_episodes = 0
    for directory in dataset_dir:
        num_episodes += count_h5_files(directory)
    use_one_hot_task = args_dict['use_onehot']

    arm_type = ArmType(args_dict['arm_type'])
    camera_names, qpos_dim, action_dim = get_arm_config(arm_type, args_dict=args_dict)

    tf_representation = args_dict["tf"]

    train_dataloader, val_dataloader, stats, is_sim = load_data(dataset_dir, num_episodes, args_dict['chunk_size'], 
                                                                batch_size_train, batch_size_val, camera_names, 
                                                                tf_representation, arm_type, with_torso, args_dict["no_decode"],
                                                                use_one_hot_task=use_one_hot_task)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, args_dict)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad , task_emb = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    task_emb = task_emb.cuda()
    return policy(qpos_data, image_data, action_data, is_pad,task_emb) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    set_seed(seed)

    policy = make_policy(policy_class, config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    policy, loaded_epoch = load_model_from_checkpoint(policy, ckpt_dir)

    if config['wandb']:
        wandb.init(name=config["run_name"], project=config["task_name"])

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(loaded_epoch + 1, num_epochs)):
        print(f'\nEpoch {epoch}')
        if epoch > 0.9 * num_epochs:
            # validation
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        actual_trained_epoch = epoch - loaded_epoch - 1
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*actual_trained_epoch:(batch_idx+1)*(actual_trained_epoch+1)])
        epoch_train_loss = epoch_summary['loss']

        if config['wandb']:
            dicts = {
                "Training loss": epoch_train_loss,
                "Epoch": epoch, 
            }
            if epoch > 0.9 * num_epochs:
                dicts["Validation loss"] = epoch_val_loss
            wandb.log(dicts)

        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            model_pth = os.path.join(ckpt_dir, f'policy_model_epoch_{epoch}_seed_{seed}.pth')
            torch.save(policy, model_pth)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    model_pth = os.path.join(ckpt_dir, f'policy_model_last.pth')
    torch.save(policy, model_pth)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        log_train_values = np.log(np.array(train_values) + 1e-10)
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), log_train_values, label='train')
        if len(validation_history)>0:
            val_values = [summary[key].item() for summary in validation_history]
            log_val_values = np.log(np.array(val_values) + 1e-10)
            plt.plot(np.linspace(0.9 * num_epochs, num_epochs-1, len(validation_history)), log_val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        # plt.close()
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = get_parser()
    main(vars(parser.parse_args()))