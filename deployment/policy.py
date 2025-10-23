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

from deployment.utils import ResizeCenterCropper
from deployment.models import EarlyFuseCNNEncoder, VecQc
import matplotlib.pyplot as plt
import lpips
import abc
from deployment.rnn_state_encoder import build_rnn_state_encoder
from deployment.models import GaussianNet, CriticHead
import threading
import time

plot_data = {
    "observations": None,
    "action": None,
    "in_gazebo": None,
    "Qc": None,
    "com_lpips": None,
    "updated": False  # Flag to indicate new data
}

def update_plot_data(observations, action, in_gazebo, Qc, com_lpips):
    plot_data["observations"] = observations
    plot_data["action"] = action
    plot_data["in_gazebo"] = in_gazebo
    plot_data["Qc"] = Qc
    plot_data["com_lpips"] = com_lpips
    plot_data["updated"] = True  # Set flag to indicate new data

def plot_thread():
    while True:
        if plot_data["updated"]:  # Only plot when new data is available
            plot_data["updated"] = False  # Reset flag

            # Extract latest data
            observations = plot_data["observations"]
            action = plot_data["action"]
            in_gazebo = plot_data["in_gazebo"]
            Qc = plot_data["Qc"]
            com_lpips = plot_data["com_lpips"]

            if observations is not None and action is not None:
                plot_results(observations, action, in_gazebo, Qc, com_lpips)

        time.sleep(0.1)  # Small delay to avoid excessive CPU usage

def plot_results(observations, action, in_gazebo, Qc, com_lpips):
    with torch.no_grad():
        pred_Q = Qc(observations['depth'], action, in_gazebo=in_gazebo)
    depth_image = Qc.depth_image
    vec = Qc.vec

    cur = observations['rgb'][0].cpu().numpy()
    goal = observations['imagegoal_sensor_v2'][0].cpu().numpy()

    if in_gazebo:
        score = com_lpips(cur, goal).item()
    else:
        score = 0

    depth_image = depth_image.permute(0, 2, 3, 1)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0][0].imshow(cur)
    axes[0][0].axis("off")
    axes[0][0].set_title("current")

    axes[0][1].imshow(goal)
    axes[0][1].axis("off")
    axes[0][1].set_title("goal")

    axes[1][0].imshow(depth_image[0].cpu().numpy(), cmap='gray')
    axes[1][0].set_title("depth")
    axes[1][0].set_xlabel("Angular vel:{:.2f}(+ is left), Linear vel: {:.2f} \n "
                          "Qc:{:.2f},Score:{:.2f}".format(action.flatten()[0].item(),
                                                          action.flatten()[1].item(),
                                                          pred_Q.item(), score))

    cbar_ax = fig.add_axes([0.55, 0.05, 0.35, 0.02])
    cmap = plt.get_cmap("gray")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, orientation='horizontal')

    axes[1][1].imshow(vec.cpu().numpy().reshape(1, -1), vmin=0, vmax=1, cmap=cmap,
                      extent=[0, depth_image.shape[-1], -2, -1])
    axes[1][1].set_title("Distance")

    plt.show()

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)

class NavNet(torch.nn.Module):
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
            resnet_baseplanes,
            normalize_visual_inputs,
            obs_transform=ResizeCenterCropper(size=(128, 128)),
            visual_encoder_embedding_size=512,
            visual_obs_inputs=['*'],
            **kwargs
    ):
        super().__init__()
        self.prev_action_embedding = nn.Linear(action_space.shape[0], 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action
        ObsEncoder = EarlyFuseCNNEncoder

        self._goal_sensor_uuid = 'imagegoal_sensor_v2'
        self.goal_visual_encoder = None
        self._hidden_size = hidden_size
        visual_observation_space = copy.deepcopy(observation_space)

        self.visual_encoder = ObsEncoder(
            visual_observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=obs_transform,
            visual_encoder_embedding_size=visual_encoder_embedding_size,
            visual_obs_inputs=visual_obs_inputs,
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

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

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
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state

class Policy(abc.ABC):
    action_distribution: nn.Module

    def __init__(self):
        pass

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return 0

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass

class NetPolicy(nn.Module, Policy):
    aux_loss_modules: nn.ModuleDict

    def __init__(
        self, net, action_space, policy_config=None, aux_loss_config=None
    ):
        super().__init__()
        self.net = net
        self.dim_actions = 2

        self.action_distribution = GaussianNet(
            self.net.output_size,
            self.dim_actions,
            policy_config.action_dist,
        )
        self.critic = CriticHead(self.net.output_size)

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return self.net.num_recurrent_layers

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    @property
    def policy_components(self):
        return (self.net, self.critic, self.action_distribution)

    def policy_parameters(self) -> Iterable[torch.Tensor]:
        for c in self.policy_components:
            yield from c.parameters()

    def all_policy_tensors(self) -> Iterable[torch.Tensor]:
        yield from self.policy_parameters()
        for c in self.policy_components:
            yield from c.buffers()

    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        return {k: v.parameters() for k, v in self.aux_loss_modules.items()}

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass

class SafeNavNetPolicy(NetPolicy):
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
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        **kwargs
    ):
        policy_config = config.habitat_baselines.rl.policy
        policy_config.action_distribution_type = "gaussian"
        super().__init__(
            NavNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
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
        self.cost_critic = copy.deepcopy(self.critic)
        self.Qc = VecQc(action_space)
        self.loss_fn = lpips.LPIPS(net='alex', verbose=False)

        plotting_thread = threading.Thread(target=plot_thread, daemon=True)
        plotting_thread.start()

        self.prev_dir = 0

    @classmethod
    def from_config(
        cls,
        config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        ppo_config = config.habitat_baselines.rl.ppo
        obs_transform = None

        return cls(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=ppo_config.hidden_size,
            rnn_type=ppo_config.rnn_type,
            num_recurrent_layers=ppo_config.num_recurrent_layers,
            backbone=ppo_config.backbone,
            resnet_baseplanes=32,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            obs_transform=obs_transform,
            visual_encoder_embedding_size=ppo_config.visual_encoder_embedding_size,
            visual_obs_inputs=ppo_config.visual_obs_inputs,
        )

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
            plot=False,
            in_gazebo=False,
            safe_correction=False
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

        pred_Q = self.Qc(observations['depth'], action, in_gazebo=in_gazebo)
        if safe_correction:
            action = self.USL_correction(observations['depth'], action, in_gazebo=in_gazebo)

        if plot:
            update_plot_data(observations, action, in_gazebo, self.Qc, self.com_lpips)

        action_log_probs = distribution.log_probs(action)
        return (value.detach(), cost_value.detach()), action.detach(), action_log_probs.detach(), rnn_hidden_states.detach(), pred_Q.detach(), self.Qc.vec.detach()

    def com_lpips(self, img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            im_1 = (img1 - 0.5) * 2
            im_2 = (img2 - 0.5) * 2
        else:
            img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
            img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
            im_1 = (img1 / 255 - 0.5) * 2
            im_2 = (img2 / 255 - 0.5) * 2
        d = self.loss_fn(im_1.cuda(), im_2.cuda())
        return d

    def USL_correction(self, obs, action, Niter=5, delta=0.5, in_gazebo=False):
        pred = self.Qc(obs, action, in_gazebo=in_gazebo).item()
        if pred <= delta:
            self.prev_dir = 0
            return action

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
            action[:, 0] += 0.3 * dir # 0.4
            action[:, 1] -= 0.3 # 0.2

            pred = self.Qc(obs, action, in_gazebo=in_gazebo).item()
            print('pred:', pred, 'action:', action)
            if pred <= delta:
                break

        if pred > delta:
            action[:, 1] = -1
        print('--------------------------------------------')
        return torch.clamp(action, min=-1, max=1).detach()

    def get_cost_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.cost_critic(features)




