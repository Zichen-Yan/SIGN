#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements tasks and measurements needed for training and benchmarking of
``habitat.Agent`` inside ``habitat.Env``.
"""

from typing import Any, Dict, Iterable, List, Optional, Union
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Observations, SensorSuite, Simulator
from habitat.core.embodied_task import EmbodiedTask, Metrics

from habitat.core.registry import registry
from habitat.config import Config, read_write

def merge_sim_episode_config(sim_config: Config, episode: Episode) -> Any:
    with read_write(sim_config):
        sim_config.scene = episode.scene_id
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.agents[sim_config.default_agent_id]
        agent_cfg = getattr(sim_config, agent_name)
        with read_write(agent_cfg):
            agent_cfg.start_position = episode.start_position
            agent_cfg.start_rotation = episode.start_rotation
            agent_cfg.is_set_start_state = True
    return sim_config

@registry.register_task(name="Nav-Drone")
class NavigationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Episode) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

    def step(self, action: Dict[str, Any], episode: Episode):
        action_name = action["action"]
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        observations: Any = {}
        if isinstance(action_name, tuple):  # there are multiple actions
            for i, a_name in enumerate(action_name):
                self._step_single_action(
                    observations,
                    a_name,
                    action,
                    episode,
                    i == len(action_name) - 1,
                )
        else:
            # get agent obs
            self._step_single_action(
                observations, action_name, action, episode
            )
        # get task obs
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )
        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )
        return observations
