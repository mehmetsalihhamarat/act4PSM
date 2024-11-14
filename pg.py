import os
import glob
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange


pth_file = 'ckpt/pick_apple/policy_model.pth'
policy = torch.load(pth_file)
print(policy)
