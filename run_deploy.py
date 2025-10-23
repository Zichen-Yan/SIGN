#!/usr/bin/env python3
import os.path
import random
import numpy as np
import torch
from deployment.trainer import OrinTrainer
import yaml
from pathlib import Path


class ConfigObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))  # Recursively convert dict to object
            else:
                setattr(self, key, value)
def main(add_control, add_lpips, fix_obs, safe_correction, plot, take_snapshot, video, hd_video):
    current_dir = Path(__file__).parent
    yaml_path = "TBD"
    with open(os.path.join(current_dir, yaml_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = ConfigObject(config)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    trainer = OrinTrainer(config, add_control, add_lpips, fix_obs, safe_correction, plot, take_snapshot, video, hd_video)
    trainer.eval()


if __name__ == "__main__":
    main(add_control=True, add_lpips=True, fix_obs=False, safe_correction=True, plot=True, take_snapshot=False, video=True, hd_video=True)
