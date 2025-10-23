#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)

from habitat_baselines.common.rollout_storage import RolloutStorage


class CostRolloutStorage(RolloutStorage):
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        super().__init__(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers,
            action_shape,
            is_double_buffered,
            discrete_actions
        )

        self.buffers["costs"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["prev_costs"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["cost_value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["cost_returns"] = torch.zeros(numsteps + 1, num_envs, 1)

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        cost_value_preds=None,
        rewards=None,
        costs=None,
        prev_costs=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            cost_value_preds=cost_value_preds,
            rewards=rewards,
            costs=costs,
            prev_costs=prev_costs
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def compute_cost_returns(self, next_cost_value, use_gae, gamma, tau):
        if use_gae:
            assert isinstance(self.buffers["cost_value_preds"], torch.Tensor)
            self.buffers["cost_value_preds"][
                self.current_rollout_step_idx
            ] = next_cost_value
            gae = 0.0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["costs"][step]
                    + gamma
                    * self.buffers["cost_value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["cost_value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["cost_returns"][step] = (  # type: ignore
                    gae + self.buffers["cost_value_preds"][step]  # type: ignore
                )

        else:
            self.buffers["cost_returns"][self.current_rollout_step_idx] = next_cost_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["cost_returns"][step] = (
                    gamma
                    * self.buffers["cost_returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["costs"][step]
                )

    def recurrent_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
        cost_advantages=None,
        train_for_mrq=False,
    ) -> Iterator[DictTree]:
        assert isinstance(self.buffers["returns"], torch.Tensor)
        num_environments = self.buffers["returns"].size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )

        dones_cpu = (
            torch.logical_not(self.buffers["masks"])
            .cpu()
            .view(-1, self._num_envs)
            .numpy()
        )

        if num_mini_batch==-1:
            # Retrieve the full data without chunking
            curr_slice = (slice(0, self.current_rollout_step_idx),
                          slice(0, num_environments))
            # Get the entire batch (all environments and all time steps)
            batch = self.buffers[curr_slice]
            # Add additional data (advantages, cost_advantages)
            if advantages is not None:
                batch["advantages"] = advantages[curr_slice]
            if cost_advantages is not None:
                batch["cost_advantages"] = cost_advantages[curr_slice]

            # Recurrent hidden states - take the first timestep for all environments
            batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]

            # Flatten the entire batch (combining time steps and environments into a single batch dimension)
            batch.map_in_place(lambda v: v.flatten(0, 1))

            # Build RNN sequence information for all environments
            batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                device=self.device,
                build_fn_result=build_pack_info_from_dones(
                    dones_cpu[0: self.current_rollout_step_idx, :num_environments  # Consider all environments
                    ].reshape(-1, num_environments),
                ),
            )

            # Yield the full batch as a tree structure
            yield batch.to_tree()
        else:
            if num_environments % num_mini_batch != 0:
                warnings.warn(
                    "Number of environments ({}) is not a multiple of the"
                    " number of mini batches ({}).  This results in mini batches"
                    " of different sizes, which can harm training performance.".format(
                        num_environments, num_mini_batch
                    )
                )

            for inds in torch.randperm(num_environments).chunk(num_mini_batch):
                curr_slice = (slice(0, self.current_rollout_step_idx), inds)
                batch = self.buffers[curr_slice]
                if advantages is not None:
                    batch["advantages"] = advantages[curr_slice]
                if cost_advantages is not None:
                    batch["cost_advantages"] = cost_advantages[curr_slice]
                batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]

                if train_for_mrq:
                    batch.map_in_place(lambda v: v.permute(1, 0, *range(2, v.ndim)).reshape(-1, *v.shape[2:]))
                else:
                    batch.map_in_place(lambda v: v.flatten(0, 1))

                if train_for_mrq:
                    batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                        device=self.device,
                        build_fn_result=build_pack_info_from_dones(
                            dones_cpu[0 : self.current_rollout_step_idx, inds.numpy()].T.reshape(-1, len(inds)),
                        ),
                    )
                else:
                    batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                        device=self.device,
                        build_fn_result=build_pack_info_from_dones(
                            dones_cpu[0: self.current_rollout_step_idx, inds.numpy()].reshape(-1, len(inds)),
                        ),
                    )

                yield batch.to_tree()

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)