#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from typing import Dict, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat.utils import profiling_wrapper
from src.custom_rollout_storage import CostRolloutStorage
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)

from src.transforms import RandomShiftsAug
from src.utils import TwoHot

EPS_PPO = 1e-5

def masked_mse(x: torch.Tensor, y: torch.Tensor, mask = None):
    if mask is None:
        return F.mse_loss(x, y)
    return (F.mse_loss(x, y, reduction='none') * mask).mean()

class DA_PPO(PPO):
    def add_transform(self, pad=4):
        self.aug = RandomShiftsAug(pad=pad)
        self.aug_weight=0.5

    def add_config(self, config):
        self.config = config
        self.use_aug_flag = True

    def add_mrq(self):
        self.two_hot = TwoHot(self.device, -5, 5, 65)
        self.dyn_weight = 1.0
        self.reward_weight = 0.1
        self.done_weight = 0.1

    def ema_update(self, target_net, online_net, tau=0.005):
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * online_param.data)

    def perturb_factor(self, step):
        return min(max(self.min_perturb_factor, 1 - self.perturb_rate * self.dormant_ratio), self.max_perturb_factor)

    def update(
        self,
        rollouts: CostRolloutStorage,
    ) -> Dict[str, float]:

        advantages = self.get_advantages(rollouts)
        learner_metrics = collections.defaultdict(list)

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        # train encoder
        if self.config.habitat_baselines.rl.data_augmentation.mrq and self.use_aug_flag:
            for epoch in range(10):
                data_generator = rollouts.recurrent_generator(advantages, num_mini_batch=2, train_for_mrq=True)
                for _bid, batch in enumerate(data_generator):
                    self._set_grads_to_none()
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _,
                        aux_loss_state,
                    ) = self._evaluate_actions(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["actions"],
                        batch["rnn_build_seq_info"],
                    )
                    # -----------------------------------------------------
                    x = aux_loss_state['visual_feats']
                    with torch.no_grad():
                        target = self.actor_critic.mrq_target.zs(x[1:])

                    pred_zs = self.actor_critic.mrq.zs(x[:-1])
                    prev_not_done = batch["masks"][1:]

                    pred_d, pred_zs, pred_r = self.actor_critic.mrq.model_all(pred_zs, batch["actions"][:-1])

                    # Mask out states past termination.
                    dyn_loss = masked_mse(pred_zs, target, prev_not_done)
                    reward_loss = (self.two_hot.cross_entropy_loss(pred_r, batch["rewards"][:-1]) * prev_not_done).mean()
                    done_loss = masked_mse(pred_d, torch.logical_not(prev_not_done).float())

                    pred_loss = self.dyn_weight * dyn_loss + self.reward_weight * reward_loss + self.done_weight * done_loss

                    pred_loss.backward()

                    self.before_step()
                    self.optimizer.step()
                    self.after_step()

            self.ema_update(self.actor_critic.mrq_target, self.actor_critic.mrq, tau=0.005)

        # train RL
        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)

            for _bid, batch in enumerate(data_generator):
                self._set_grads_to_none()
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    aux_loss_state,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                    batch["rnn_build_seq_info"],
                )
                if isinstance(values, tuple):
                    values, cost_values = values

                if isinstance(dist_entropy, tuple):
                    dist, dist_entropy = dist_entropy

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])

                surr1 = batch["advantages"] * ratio
                surr2 = batch["advantages"] * (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                )
                action_loss = -torch.min(surr1, surr2)

                values = values.float()
                orig_values = values

                if self.use_clipped_value_loss:
                    delta = values.detach() - batch["value_preds"]
                    value_pred_clipped = batch["value_preds"] + delta.clamp(-self.clip_param, self.clip_param)

                    values = torch.where(
                        delta.abs() < self.clip_param,
                        values,
                        value_pred_clipped,
                    )

                value_loss = 0.5 * F.mse_loss(values, batch["returns"], reduction="none")

                action_loss, value_loss, dist_entropy = map(
                    torch.mean,
                    (action_loss, value_loss, dist_entropy),
                )
                all_losses = [self.value_loss_coef * value_loss, action_loss]

                if isinstance(self.entropy_coef, float):
                    all_losses.append(-self.entropy_coef * dist_entropy)
                else:
                    all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))
                # -----------------------------------------------------
                if self.config.habitat_baselines.rl.data_augmentation.shift:
                    for key in ['rgb']: # , 'imagegoal_sensor_v2'
                        data = batch['observations'][key] /255.0
                        data = data.permute(0, 3, 1, 2)
                        aug_data = self.aug(data)
                        aug_data = aug_data.permute(0, 2, 3, 1) * 255
                        batch['observations'][key]=aug_data

                    (
                        values_aug,
                        action_log_probs_aug,
                        dist_entropy_aug,
                        _,
                        aux_loss_state,
                    ) = self._evaluate_actions(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["actions"],
                        batch["rnn_build_seq_info"],
                    )
                    if isinstance(values_aug, tuple):
                        values_aug, cost_values_aug = values_aug

                    if isinstance(dist_entropy_aug, tuple):
                        dist_aug, dist_entropy_aug = dist_entropy_aug

                    # Compute Augmented Loss
                    action_loss_aug = - action_log_probs_aug.mean() # KL simplified
                    value_loss_aug = .5 * (torch.detach(values) - values_aug).pow(2).mean()
                    aug_loss = value_loss_aug + action_loss_aug
                    all_losses.append(self.aug_weight * aug_loss)

                total_loss = torch.stack(all_losses).sum()

                total_loss.backward()
                grad_norm = self.before_step()
                self.optimizer.step()
                self.after_step()

                with inference_mode():
                    if self.config.habitat_baselines.rl.data_augmentation.mrq and self.use_aug_flag:
                        learner_metrics['mrq_r_loss'].append(reward_loss)
                        learner_metrics['mrq_s_loss'].append(dyn_loss)
                        learner_metrics['mrq_d_loss'].append(done_loss)

                    if "is_coeffs" in batch:
                        record_min_mean_max(
                            batch["is_coeffs"], "ver_is_coeffs"
                        )
                    record_min_mean_max(orig_values, "value_pred")
                    record_min_mean_max(ratio, "prob_ratio")

                    learner_metrics["value_loss"].append(value_loss)
                    learner_metrics["action_loss"].append(action_loss)
                    learner_metrics["dist_entopy"].append(dist_entropy)
                    if self.config.habitat_baselines.rl.data_augmentation.shift:
                        learner_metrics['aug_loss'].append(aug_loss)
                        learner_metrics['aug_act_loss'].append(action_loss_aug)
                        learner_metrics['aug_critic_loss'].append(value_loss_aug)

                    if epoch == (self.ppo_epoch - 1):
                        learner_metrics["ppo_fraction_clipped"].append(
                            (ratio > (1.0 + self.clip_param)).float().mean()
                            + (ratio < (1.0 - self.clip_param)).float().mean()
                        )

                    learner_metrics["grad_norm"].append(grad_norm)
                    if isinstance(
                        self.entropy_coef, LagrangeInequalityCoefficient
                    ):
                        learner_metrics["entropy_coef"].append(
                            self.entropy_coef().detach()
                        )

            profiling_wrapper.range_pop()  # PPO.update epoch

        if hasattr(self.actor_critic, 'Qc'):
            data_generator = rollouts.recurrent_generator(advantages, 1)
            for _bid, batch in enumerate(data_generator):
                cost = batch['costs'].flatten()

                zeros = torch.nonzero(cost == torch.tensor(0, device=cost.device)).flatten()  # Indices of zeros
                ones = torch.nonzero(cost == torch.tensor(1, device=cost.device)).flatten()  # Indices of ones

                qc_loss = torch.tensor(0, device=cost.device)
                # Find the smaller class size
                if zeros.numel() > 0 and ones.numel() > 0:
                    min_class_size = min(zeros.numel(), ones.numel())

                    for i in range(10):
                        sampled_zeros = zeros[torch.randperm(zeros.numel())[:min_class_size]]
                        sampled_ones = ones[torch.randperm(ones.numel())[:min_class_size]]

                        # Combine the sampled indices
                        sampled_indices = torch.cat([sampled_zeros, sampled_ones])

                        sampled_depths = batch["observations"]["depth"][sampled_indices]
                        sampled_actions = batch["actions"][sampled_indices]
                        hard_label = cost[sampled_indices]

                        # one-step
                        logits = self.actor_critic.Qc(sampled_depths, sampled_actions)
                        loss = nn.BCELoss()
                        depth_vector = self.actor_critic.Qc.vec
                        min_dis, _ = depth_vector.min(dim=1)
                        soft_label = nn.Sigmoid()((0.3 - min_dis) * 10)

                        n = soft_label.shape[0]   # Random boolean mask: True for soft, False for hard
                        mixed_label = torch.where(torch.rand(n, device=self.device) < 0.5, soft_label, hard_label)

                        qc_loss = loss(logits.flatten(), mixed_label) # soft_label, hard_label

                        self.actor_critic.Qc_opt.zero_grad()
                        qc_loss.backward()
                        self.actor_critic.Qc_opt.step()

            all_depths = batch["observations"]["depth"]
            all_actions = batch["actions"]
            all_costs = batch['costs'].flatten()
            with torch.no_grad():
                logits = self.actor_critic.Qc(all_depths, all_actions)
                loss = nn.BCELoss()
                qc_loss = loss(logits.flatten(), all_costs)

            learner_metrics['qc_loss'].append(qc_loss)

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    def get_cost_advantages(self, rollouts: CostRolloutStorage):
        advantages = (
            rollouts.buffers["cost_returns"]  # type: ignore
            - rollouts.buffers["cost_value_preds"]
        )
        if not self.use_normalized_advantage:
            return advantages

        var, mean = self._compute_var_mean(
            advantages[torch.isfinite(advantages)]
        )

        advantages -= mean

        return advantages.mul_(torch.rsqrt(var + EPS_PPO))


