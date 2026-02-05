import pathlib

### Task parameters
DATA_DIR = '/home/nural/IsaacLab/datasets'   # Dataset dırectıon
TASK_CONFIGS = {
    'lift_needle':{
        'dataset_dir': DATA_DIR + '/distillation',        # File name that contains dataset in
        'num_episodes': 50,                             # This includes 0 as well
        'camera_names': ['rgb_endo_cam','rgb_wrist_cam']                 # Put camera names
    },
}
