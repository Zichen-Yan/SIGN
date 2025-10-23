#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from habitat import Config, logger
from habitat.config import read_write
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from src.transforms import apply_obs_transforms_batch
# from habitat_baselines.common.rollout_storage import RolloutStorage
from src.custom_rollout_storage import CostRolloutStorage
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)

from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.rl.ppo import PPO
from src.algo.PPO import DA_PPO

from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_num_actions,
    inference_mode,
    is_continuous_action_space,
)
from src.visualization import observations_to_image
from src.transforms import ResizeCenterCropper

@baseline_registry.register_trainer(name="SIGN")
class PPOTrainer(BaseRLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = [ResizeCenterCropper(size=(128, 128), channels_last=True)]

        self._encoder = None
        self._obs_space = None

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        name = self.config.habitat_baselines.rl.policy.name
        policy = baseline_registry.get_policy(name)
        observation_space = self.obs_space
        observation_space = apply_obs_transforms_obs_space(observation_space, self.obs_transforms)
        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=self.orig_policy_action_space,
        )

        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if ppo_cfg.agent_name=="PPO":
            self.agent = PPO.from_config(self.actor_critic, ppo_cfg)
        elif ppo_cfg.agent_name=="DA_PPO":
            self.agent = DA_PPO.from_config(self.actor_critic, ppo_cfg)
            self.agent.add_config(self.config)
            if self.config.habitat_baselines.rl.data_augmentation.shift:
                self.agent.add_transform(pad=self.config.habitat_baselines.rl.data_augmentation.pad)
            if self.config.habitat_baselines.rl.data_augmentation.mrq:
                self.agent.add_mrq()
        else:
            raise NotImplementedError

    def _init_envs(self, config=None, is_eval: bool = False):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            workers_ignore_signals=False,
            enforce_scenes_greater_eq_environments=is_eval,
        )

    def _init_train(self):
        if self.config.habitat_baselines.verbose:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        self._init_envs()

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]
        if is_continuous_action_space(action_space):
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            action_shape = (1,)
            discrete_actions = True

        ppo_cfg = self.config.habitat_baselines.rl.ppo

        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.habitat_baselines.checkpoint_folder):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        self.rollouts = CostRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_costs = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
            cost=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.ac_infer_time = 0.0
        self.update_stat_time = 0.0
        self.update_model_time = 0.0
        self.t_start = time.time()

        self.prev_costs = torch.zeros(self.envs.num_envs, 1)

    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            # "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, file_name
            ),
        )
        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, "latest.pth"
            ),
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if not isinstance(k, str) or k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if isinstance(subk, str)
                        and k + "." + subk not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self):
        num_envs = self.envs.num_envs
        env_slice = slice(0, num_envs)

        t_sample_action = time.time()

        # sample actions
        with inference_mode():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[0],
                env_slice,
            ]
            step_batch.map_in_place(lambda x: x.to(self.device))
            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )
            
        if isinstance(values, tuple):
            values, cost_values = values
        else:
            cost_values = None

        self.pth_time += time.time() - t_sample_action
        self.ac_infer_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.cpu().unbind(0)
        ):
            if is_continuous_action_space(self.policy_action_space):
                # Clipping actions to the specified limits
                act = np.clip(
                    act.numpy(),
                    self.policy_action_space.low,
                    self.policy_action_space.high,
                )
            else:
                act = act.item()
            self.envs.async_step_at(index_env, act)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            cost_value_preds=cost_values,
            buffer_index=0,
        )

    def _collect_environment_result(self):
        num_envs = self.envs.num_envs
        env_slice = slice(0, num_envs)

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        costs = torch.tensor([d['collisions.is_collision'] for d in infos], dtype=torch.float).view(-1,1).to(self.current_episode_reward.device)
        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        self.current_episode_costs[env_slice] += costs
        current_ep_reward = self.current_episode_reward[env_slice]
        current_ep_cost = self.current_episode_costs[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["cost"][env_slice] += current_ep_cost.where(done_masks, current_ep_cost.new_zeros(()))
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)
        self.current_episode_costs[env_slice].masked_fill_(done_masks, 0.0)

        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            costs=costs,
            prev_costs=self.prev_costs,
            next_masks=not_done_masks,
            buffer_index=0,
        )
        self.prev_costs = costs
        self.rollouts.advance_rollout(0)

        self.pth_time += time.time() - t_update_stats
        self.update_stat_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.habitat_baselines.rl.ppo
        t_update_model = time.time()
        with inference_mode():
            step_batch = self.rollouts.buffers[self.rollouts.current_rollout_step_idx]

            if hasattr(self.rollouts, "compute_returns"):
                next_value = self.actor_critic.get_value(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )
                self.rollouts.compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)

            if hasattr(self.actor_critic, "cost_critic") and hasattr(self.rollouts, "compute_cost_returns"):
                next_cost_value = self.actor_critic.get_cost_value(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )
                self.rollouts.compute_cost_returns(next_cost_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)

        self.agent.train()
        losses = self.agent.update(self.rollouts)

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model
        self.update_model_time += time.time() - t_update_model
        return losses

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack([self.running_episode_stats[k] for k in stats_ordering], 0)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        self.num_steps_done += count_steps_delta

        return losses

    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        # log stats
        if (self.num_updates_done % self.config.habitat_baselines.log_interval == 0):
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "totaltime: {:.3f} min\tupdate: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\tac-infer-time: {:.3f}s\tupdate-stat-time: {:.3f}s\tupdate-model-time: {:.3f}s\t"
                "frames: {}\tschedule: {}".format(
                    (time.time()-self.global_start_time)/60,
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.ac_infer_time,
                    self.update_stat_time,
                    self.update_model_time,
                    self.num_steps_done,
                    self.num_steps_done/self.config.habitat_baselines.total_num_steps
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        self.global_start_time = time.time()
        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if ppo_cfg.use_linear_lr_decay:
            lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=lambda x: 1 - self.percent_done(),
            )

        with (
            get_writer(
                self.config,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                self._compute_actions_and_step_envs()      # use buffer data update actor_critic

                for step in range(ppo_cfg.num_steps):                     # collect num_steps batch data
                    is_last_step = (step + 1) == ppo_cfg.num_steps

                    count_steps_delta += self._collect_environment_result()
                    profiling_wrapper.range_pop()  # _collect_rollout_step

                    if not is_last_step:
                        profiling_wrapper.range_push("_collect_rollout_step")
                        self._compute_actions_and_step_envs()

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop
                if self.num_steps_done >= int(self.config.habitat_baselines.rl.data_augmentation.stop_step):
                    self.agent.use_aug_flag=False

                losses = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(losses, count_steps_delta)
                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        safe_ckpt_dict=None
        safe_network_path = self.config.habitat_baselines.rl.policy.safety_model_path
        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.habitat_baselines.eval.should_load_ckpt:
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            if safe_network_path is not None and os.path.exists(safe_network_path):
                safe_ckpt_dict = self.load_checkpoint(safe_network_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {}

        if self.config.habitat_baselines.eval.use_ckpt_config:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.habitat_baselines.rl.ppo

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if (
            len(self.config.habitat_baselines.video_option) > 0
            and self.config.habitat_baselines.video_render_top_down
        ):
            with read_write(config):
                config.habitat.task.measurements.append("top_down_map")
                config.habitat.task.measurements.append("collisions")

        if (
            len(config.habitat_baselines.video_render_views) > 0
            and len(self.config.habitat_baselines.video_option) > 0
        ):
            with read_write(config):
                for render_view in config.habitat_baselines.video_render_views:
                    uuid = config.habitat.simulator[render_view].uuid
                    config.habitat.gym.obs_keys.append(uuid)
                    config.habitat_baselines.sensors.append(render_view)

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {config}")

        self._init_envs(config, is_eval=True)

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]
        if is_continuous_action_space(action_space):
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            action_shape = (1,)
            discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if safe_ckpt_dict is not None:
                if "state_dict" not in safe_ckpt_dict:
                    self.agent.actor_critic.Qc.load_state_dict(safe_ckpt_dict, strict=False)
                else:
                    filtered_state_dict = {k: v for k, v in safe_ckpt_dict["state_dict"].items() if "Qc" in k}
                    self.agent.load_state_dict(filtered_state_dict, strict=False)
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.config.habitat_baselines.num_environments,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        prev_costs = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.float32,
        )
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        infos = None
        rgb_frames = [[] for _ in range(self.config.habitat_baselines.num_environments)]  # type: List[List[np.ndarray]]
        if len(self.config.habitat_baselines.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self.actor_critic.eval()

        while (len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0):
            current_episodes_info = self.envs.current_episodes()
            batch['cost'] = prev_costs
            (
                _,
                actions,
                _,
                test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=True,
                eval=False
            )

            prev_actions.copy_(actions)
            if is_continuous_action_space(self.policy_action_space):
                step_data = [
                    np.clip(a.numpy(),
                        self.policy_action_space.low,
                        self.policy_action_space.high,
                    )
                    for a in actions.cpu()
                ]
            else:
                step_data = [a.item() for a in actions.cpu()]

            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if len(self.config.habitat_baselines.video_option) > 0 and infos is not None:
                    frame = observations_to_image({k: v[i] for k, v in batch.items()}, infos[i])
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )
                    if self.config.habitat_baselines.video_render_all_info:
                        frame = overlay_frame(frame, infos[i])
                    rgb_frames[i].append(frame)

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            prev_costs = torch.vstack([torch.tensor(info['collisions.is_collision'], dtype=torch.float32, device=self.device) for info in infos])

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float, device=self.device).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ] 
                    == evals_per_ep
                ):
                     envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if len(self.config.habitat_baselines.video_option) > 0:
                        generate_video(
                            video_option=self.config.habitat_baselines.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )
                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()