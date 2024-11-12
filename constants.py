import pathlib

### Task parameters
DATA_DIR = 'data'
TASK_CONFIGS = {
    'pick_basket':{
        'dataset_dir': DATA_DIR + '/pick_basket',
        'num_episodes': 50,
        'episode_len': 350,
        'camera_names': ['rgb_head', 'rgb_left_hand', 'rgb_right_hand']
    },
}
