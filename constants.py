import pathlib

### Task parameters
DATA_DIR = 'data'
TASK_CONFIGS = {
    'pick_basket':{
        'dataset_dir': DATA_DIR + '/pick_basket',
        'num_episodes': 50,
        'camera_names': ['rgb_head', 'rgb_left_hand', 'rgb_right_hand']
    },

    'pick_apple':{
        'dataset_dir': DATA_DIR + '/pick_apple',
        'num_episodes': 50,
        'camera_names': ['rgb_head', 'rgb_left_hand', 'rgb_right_hand']
    },

    'pick_carrot':{
        'dataset_dir': DATA_DIR + '/pick_carrot',
        'num_episodes': 62,
        'camera_names': ['rgb_head', 'rgb_left_hand', 'rgb_right_hand']
    },
}
