import subprocess
import json

# Load config from file
with open("act_train_cfg.json", "r") as f:
    config = json.load(f)

# Base command
args = ["python", "imitate_episodes.py"]

args.extend(["--ckpt_dir", "ckpt/" + config["file_name"]])
args.extend(["--policy_class", config["policy_class"]])
args.extend(["--task_name", config["task_name_act"]])  # <-- also fix this, you need task_name_act
args.extend(["--batch_size", str(config["batch_size"])])
args.extend(["--seed", str(config["seed"])])
args.extend(["--num_epochs", str(config["num_epochs"])])
args.extend(["--lr", str(config["lr"])])
args.extend(["--kl_weight", str(config["kl_weight"])])
args.extend(["--chunk_size", str(config["chunk_size"])])
args.extend(["--hidden_dim", str(config["hidden_dim"])])
args.extend(["--dim_feedforward", str(config["dim_feedforward"])])

# Correct boolean check
if config["temporal_agg"] == True:
    args.append("--temporal_agg")  # No value after flags like --temporal_agg

# Run the command
subprocess.run(args)
