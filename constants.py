import pathlib

### Task parameters
DATA_DIR = '/home/nural/IsaacLab/datasets'
TASK_CONFIGS = {
    'lift_needle':{
        'dataset_dir': DATA_DIR + '/Feb_24_2025',
        'num_episodes': 49,
        'camera_names': ['rgb_endo_cam', 'rgb_wrist_cam']
    },
}
