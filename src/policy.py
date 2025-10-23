from email.policy import default
from nis import match
from typing import (
    Dict,
    Iterable,
    Optional,
    Union,
    List,
    Tuple
)
import copy
import torch
import numpy as np
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from habitat import logger
from habitat.config import Config
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import NetPolicy, Net, Policy

from habitat_baselines.utils.common import get_num_actions, is_continuous_action_space
from habitat_baselines.common.baseline_registry import baseline_registry


from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoalSensor,
    InstanceImageGoalHFOVSensor
)
from src.sensor import (
    GibsonImageGoalFeatureSensor,
    ImageGoalSensorV2,
    QueriedImageSensor,
)

from src.transforms import (
    ResizeCenterCropper,
    ResizeRandomCropper,
)
from src.models import EarlyFuseCNNEncoder, VecQc
from src.models import CustomGaussianNet
import matplotlib.pyplot as plt
import lpips
from src.models import MrQHead

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class NavNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            hidden_size,
            num_recurrent_layers,
            rnn_type,
            backbone,
            goal_backbone,
            resnet_baseplanes,
            normalize_visual_inputs,
            obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa: B008
            visual_encoder_embedding_size=512,
            visual_obs_inputs=['*'],
            **kwargs
    ):
        super().__init__()

        if is_continuous_action_space(action_space):
            self.prev_action_embedding = nn.Linear(action_space.shape[0], 32)
        else:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action
        ObsEncoder = EarlyFuseCNNEncoder

        logger.info('Type of observation encoder: {}'.format(ObsEncoder))
        # tied_param = {}

        # construct goal encoder
        logger.info(f'Create goal encoder : {goal_backbone}')
        if GibsonImageGoalFeatureSensor.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = GibsonImageGoalFeatureSensor.cls_uuid
        elif ImageGoalSensorV2.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = ImageGoalSensorV2.cls_uuid
        elif QueriedImageSensor.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = QueriedImageSensor.cls_uuid
        elif InstanceImageGoalSensor.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = InstanceImageGoalSensor.cls_uuid
        else:
            raise ValueError(f"Wrong goal sensor")

        self.goal_visual_encoder = None
        self._hidden_size = hidden_size
        logger.info('Create visual inputs encoder')
        visual_observation_space = copy.deepcopy(observation_space)

        self.visual_encoder = ObsEncoder(
            visual_observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=obs_transform,
            visual_encoder_embedding_size=visual_encoder_embedding_size,
            visual_obs_inputs=visual_obs_inputs
        )

        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(
                np.prod(self.visual_encoder.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )

        self.state_encoder = build_rnn_state_encoder(
            self._hidden_size + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def perception_embedding_size(self):
        self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info=None):
        x = []
        aux_loss_state = {}
        # forward visual encoder
        if "visual_features" in observations:
            visual_feats = observations["visual_features"]
        else:
            visual_feats, _ = self.visual_encoder(observations)

        visual_feats = self.visual_fc(visual_feats)
        x.append(visual_feats)

        # forward goal encoder
        if self.goal_visual_encoder != None:
            goal_image = observations[self._goal_sensor_uuid]
            goal_output, _ = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        if isinstance(self.prev_action_embedding, nn.Linear):
            prev_actions = self.prev_action_embedding(
                ((prev_actions.float()) * masks).squeeze(dim=-1)
            )
        else:
            prev_actions = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
            )
        x.append(prev_actions)

        x = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks, rnn_build_seq_info)
        aux_loss_state["visual_feats"] = visual_feats

        return out, rnn_hidden_states, aux_loss_state

@baseline_registry.register_policy
class NavNetPolicy(NetPolicy):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="GRU",
        resnet_baseplanes=32,
        backbone="resnet18",
        goal_backbone="clip",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa : B008
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        **kwargs
    ):
        policy_config=None
        if is_continuous_action_space(action_space):
            policy_config=config.habitat_baselines.rl.policy
            policy_config.defrost()
            policy_config.action_distribution_type="gaussian"
            policy_config.freeze()

        super().__init__(
            NavNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                goal_backbone=goal_backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                visual_encoder_embedding_size=visual_encoder_embedding_size,
                visual_obs_inputs=visual_obs_inputs,
                **kwargs
            ),
            action_space,
            policy_config
        )

        if is_continuous_action_space(action_space):
            self.action_distribution = CustomGaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.action_dist,
            )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        eval=False
    ):
        with torch.no_grad():
            features, rnn_hidden_states, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
            distribution = self.action_distribution(features)
            value = self.critic(features)

            if deterministic:
                action = distribution.mode()
            else:
                action = distribution.sample()

            action_log_probs = distribution.log_probs(action)
        return value, action, action_log_probs, rnn_hidden_states

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        ppo_config = config.habitat_baselines.rl.ppo

        return cls(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=ppo_config.hidden_size,
            rnn_type=ppo_config.rnn_type,
            num_recurrent_layers=ppo_config.num_recurrent_layers,
            backbone=ppo_config.backbone,
            goal_backbone=ppo_config.goal_backbone,
            resnet_baseplanes=32,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            obs_transform=None,
            visual_encoder_embedding_size=ppo_config.visual_encoder_embedding_size,
            visual_obs_inputs=ppo_config.visual_obs_inputs,
            visual_encoder_init=None,
            task_type_embed=ppo_config.task_type_embed,
            task_type_embed_size=ppo_config.task_type_embed_size
        )

@baseline_registry.register_policy
class SafeNavNetPolicy(NavNetPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="GRU",
        resnet_baseplanes=32,
        backbone="resnet18",
        goal_backbone="clip",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa : B008
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        **kwargs
    ):
        super().__init__(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                goal_backbone=goal_backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                visual_encoder_embedding_size=visual_encoder_embedding_size,
                visual_obs_inputs=visual_obs_inputs,
                **kwargs
        )
        self.cost_critic = copy.deepcopy(self.critic)
        self.Qc = VecQc(action_space)
        self.Qc_opt = torch.optim.Adam(self.Qc.parameters(), lr=1e-4)

        if kwargs['config'].habitat_baselines.rl.data_augmentation.mrq:
            self.mrq = MrQHead(action_space=action_space, state_dim=self.net._hidden_size)
            self.mrq_target = copy.deepcopy(self.mrq)

        self.prev_dir = 0
        self.correction_cnt = 0
        self.gazebo_action = None

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
            eval=False,
            in_gazebo=False
        ):
        features, rnn_hidden_states, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)
        cost_value = self.cost_critic(features)

        if deterministic:
            action = distribution.mode
        else:
            action = distribution.sample()

        self.raw_action = action.detach().clone()
        action_log_probs = distribution.log_probs(action)

        return (value.detach(), cost_value.detach()), action.detach(), action_log_probs.detach(), rnn_hidden_states.detach()

    def USL_correction(self, obs, action, Niter=5, delta=0.5, in_gazebo=False):
        pred = self.Qc(obs, action, in_gazebo=in_gazebo).item()
        if pred <= delta:
            self.prev_dir = 0
            return action

        self.correction_cnt += 1
        vec = self.Qc.vec
        left_min = vec[:, :6].min()
        right_min = vec[:, -6:].min()
        mid_min = vec[:, 3:-3].min()

        if self.prev_dir == 0:
            if (left_min > right_min):
                dir = 1  # left
            else:
                dir = -1  # right
            self.prev_dir = dir
        else:
            dir = self.prev_dir

        for i in range(Niter):
            action[:, 0] += 0.3 * dir
            action[:, 1] -= 0.3

            pred = self.Qc(obs, action, in_gazebo=in_gazebo).item()
            # print('pred:', pred, 'action:', action)
            if pred <= delta:
                break

        if pred > delta:
            action[:, 1] = -1

        return torch.clamp(action, min=-1, max=1).detach()

    def get_cost_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.cost_critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)
        cost_value = self.cost_critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return (
            (value, cost_value),
            action_log_probs,
            (distribution, distribution_entropy),
            rnn_hidden_states,
            aux_loss_state,
        )

    @property
    def policy_components(self):
        return (self.net, self.critic, self.cost_critic, self.action_distribution)

    @property
    def actor_components(self):
        return (self.net, self.action_distribution)

    def actor_parameters(self) -> Iterable[torch.Tensor]:
        for c in self.actor_components:
            yield from c.parameters()

    @property
    def critic_components(self):
        return (self.net, self.critic, self.cost_critic)

    def critic_parameters(self) -> Iterable[torch.Tensor]:
        for c in self.critic_components:
            yield from c.parameters()



