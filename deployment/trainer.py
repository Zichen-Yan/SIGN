#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Dict
import torch

from deployment.env import Env
from collections import OrderedDict
import gym
from gym.spaces import Box
from deployment.policy import SafeNavNetPolicy
import numpy as np
from deployment.utils import batch_obs
from pathlib import Path
import os
from deployment.utils import ResizeCenterCropper, apply_obs_transforms_obs_space, apply_obs_transforms_batch
from PIL import Image
import imageio
import pickle

class DA_PPO(torch.nn.Module):
    def __init__(self, actor_critic) -> None:
        super().__init__()  # Properly initialize nn.Module
        self.actor_critic = actor_critic

class DeployTrainer:
    def __init__(self, config=None, add_control=False, add_lpips=False, fix_obs=True, safe_correction=False, plot=False, take_snapshot=False, video=False, hd_video=False):
        super().__init__()
        self.config = config
        self.safe_correction = safe_correction
        self.plot = plot
        self.video = video
        self.hd_video = hd_video

        self.actor_critic = None
        self.agent = None
        self.envs = Env(config, add_control, add_lpips, fix_obs, take_snapshot)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_transforms = [ResizeCenterCropper(size=(128, 128), channels_last=True)]

    def _setup_actor_critic_agent(self) -> None:
        observation_space = self.envs.observation_spaces
        observation_space = apply_obs_transforms_obs_space(observation_space, self.obs_transforms)
        self.actor_critic = SafeNavNetPolicy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=OrderedDict([
            ('continuous_control', gym.spaces.Dict({
                'angular_velocity': Box(-1.0, 1.0, (1,), dtype='float32'),
                'linear_velocity': Box(-1.0, 1.0, (1,), dtype='float32')
            }))
        ]),
        )
        self.actor_critic.to(self.device)
        self.agent = DA_PPO(self.actor_critic)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def eval(self) -> None:
        current_dir = Path(__file__).parent

        safe_network_path = os.path.join(current_dir, "../ckpt/safety_model.pth")
        safe_ckpt_dict = self.load_checkpoint(safe_network_path, map_location="cpu")

        model_path = os.path.join(current_dir, "../ckpt/nav_model.pth")
        ckpt_dict = self.load_checkpoint(model_path, map_location="cpu")

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        action_space = self.envs.action_spaces
        self.policy_action_space = action_space

        action_shape = (2,)
        discrete_actions = False

        self._setup_actor_critic_agent()
        self.agent.load_state_dict(ckpt_dict['state_dict'], strict=False)
        if safe_ckpt_dict is not None:
            filtered_state_dict = {k: v for k, v in safe_ckpt_dict["state_dict"].items() if "Qc" in k}
            self.agent.load_state_dict(filtered_state_dict, strict=False)

        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)  # type: ignore
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            1,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            1,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        prev_costs = torch.zeros(
            1,
            1,
            device=self.device,
            dtype=torch.float32,
        )

        self.actor_critic.eval()

        cnt = 0
        start_time = time.time()
        done_flag = False

        if self.video:
            log_path = make_log_dir()

        while True:
            batch['cost'] = prev_costs
            (
                _,
                actions,
                _,
                test_recurrent_hidden_states,
                Q,
                vec
            ) = self.actor_critic.act(
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=True,
                plot=self.plot,
                in_gazebo=True,
                safe_correction=False if done_flag else self.safe_correction,
            )

            prev_actions.copy_(actions)
            step_data = [
                np.clip(a.numpy(), self.policy_action_space.low, self.policy_action_space.high,)
                for a in actions.cpu()
            ]
            outputs = self.envs.step(step_data)
            observations, dones, pos, ori = [list(x) for x in zip(*outputs)]

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            if self.video: #  and not done_flag
                if cnt == 0:
                    if self.hd_video:
                        arr = self.envs.raw_current_goal
                    else:
                        arr = np.array(batch['imagegoal_sensor_v2'][0].cpu())
                    img = Image.fromarray(arr)
                    save_path = os.path.join(log_path, "target1.png")
                    img.save(save_path)
                if cnt % 5 ==0: #  and not done_flag
                    if self.hd_video:
                        arr=self.envs.raw_rgb_image
                    else:
                        arr = np.array(batch['rgb'][0].cpu())
                    img = Image.fromarray(arr)
                    save_path = os.path.join(log_path, f"rgb_{cnt // 5}.png")
                    img.save(save_path)

                    arr = np.array(batch['depth'].squeeze().cpu())
                    depth = (arr * 65535).astype(np.uint16)

                    save_path = os.path.join(log_path, f"depth_{cnt // 5}.png")
                    imageio.imwrite(save_path, depth)

                    clipped_act = torch.clamp(actions, min=-1.0, max=1.0).cpu().numpy()
                    raw_action = torch.clamp(self.actor_critic.raw_action, min=-1, max=1).cpu().numpy()

                    record = {
                        'pos': pos[0],
                        'ori': ori[0],
                        'act': clipped_act[0],
                        'raw_act': raw_action[0],
                        'Q': Q[0].item(),
                        'vec': vec[0],
                        'done': done_flag,
                    }
                    with open(os.path.join(log_path, f"tra_data.pkl"), "ab") as f:  # 'a' means append mode
                        pickle.dump(record, f)

            if dones[0] == True and not done_flag:
                print('Success !!!!!!!!')
                done_flag = True

            cnt += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0 and not done_flag:
                freq = cnt / elapsed_time
                print(f"Frequency: {freq:.2f} Hz")

def make_log_dir(base_name="log"):
    i = 1
    while True:
        name = f"{base_name}{i}"
        path = os.path.join("deployment/logs", name)
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Log path: {name}")
            break
        i += 1
    return path


