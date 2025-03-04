import torch
from utils import load_data, set_seed

# Set seed for reproducibility
set_seed(42)

# Parameters for verification
dataset_dir = '/home/nural/IsaacLab/datasets' # Change this to your dataset path
num_episodes = 9 # count 0
camera_names = ["table_cam_RGB", "wrist_cam_RGB"]
batch_size_train = 2
batch_size_val = 2
chunk_size = 20
img_compressed = False

# Load data
train_loader, val_loader, norm_stats = load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    chunk_size,
    img_compressed
)

# Verify the dataloader outputs
for image_data, qpos_data, action_data, is_pad in train_loader:
    print(f"Image data shape: {image_data.shape}")   # Should be (batch_size, num_cameras, 3, 240, 320)
    print(f"Qpos data shape: {qpos_data.shape}")     # Should be (batch_size, 16)
    print(f"Action data shape: {action_data.shape}") # Should be (batch_size, chunk_size, 7)
    print(f"Padding mask shape: {is_pad.shape}")     # Should be (batch_size, chunk_size)
    break
