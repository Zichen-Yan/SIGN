import torch
from torch import nn as nn
from src import resnet as fast_resnet
from deployment.utils import ResizeCenterCropper
from torch import Tensor

VISUAL_SENSORS_UUID = [
    'rgb', 'depth', 'semantic', 'feature', 'imagegoal_sensor_v2', 'panoramic_rgb_sensor', 'queried_image_sensor',
    'instance_imagegoal'
]

MOD_SEP = '/'

class RunningMeanAndVar(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        assert n_channels > 0
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.zeros(()))
        self._mean: torch.Tensor = self._mean
        self._var: torch.Tensor = self._var
        self._count: torch.Tensor = self._count

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            n = x.size(0)
            x_channels_first = (
                x.transpose(1, 0).contiguous().view(x.size(1), -1)
            )
            new_mean = x_channels_first.mean(-1, keepdim=True)
            new_count = torch.full_like(self._count, n)
            new_var = (
                (x_channels_first - new_mean).pow(2).mean(dim=-1, keepdim=True)
            )

            new_mean = new_mean.view(1, -1, 1, 1)
            new_var = new_var.view(1, -1, 1, 1)

            m_a = self._var * (self._count)
            m_b = new_var * (new_count)
            M2 = (
                m_a
                + m_b
                + (new_mean - self._mean).pow(2)
                * self._count
                * new_count
                / (self._count + new_count)
            )

            self._var = M2 / (self._count + new_count)
            self._mean = (self._count * self._mean + new_count * new_mean) / (
                self._count + new_count
            )

            self._count += new_count

        inv_stdev = torch.rsqrt(
            torch.max(self._var, torch.full_like(self._var, 1e-2))
        )
        return torch.addcmul(-self._mean * inv_stdev, x, inv_stdev)

def make_resnet_encoder(make_backbone, n_input_channels, baseplanes, ngroups,
                        spatial_size=128, normalize_inputs=False,
                        after_compression_flat_size=2048, film_reduction=None, film_layers=None):
    modules = []
    if normalize_inputs:
        modules.append(RunningMeanAndVar(n_input_channels))

    if film_reduction != None:
        backbone = make_backbone(n_input_channels, baseplanes, ngroups, film_reduction, film_layers)
    else:
        backbone = make_backbone(n_input_channels, baseplanes, ngroups)
    modules.append(backbone)
    final_spatial = int(spatial_size * backbone.final_spatial_compress)
    num_compression_channels = int(
        round(after_compression_flat_size / (final_spatial ** 2))
    )
    compression = nn.Sequential(
        nn.Conv2d(
            backbone.final_channels,
            num_compression_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(1, num_compression_channels),
        nn.ReLU(True),
    )
    modules.append(compression)
    output_shape = (
        num_compression_channels,
        final_spatial,
        final_spatial,
    )
    return nn.Sequential(*modules), output_shape


class EarlyFuseCNNEncoder(nn.Module):
    def __init__(
            self,
            observation_space,
            baseplanes=32,
            ngroups=32,
            spatial_size=128,
            backbone=None,
            normalize_visual_inputs=False,
            obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa: B008
            visual_encoder_embedding_size=512,
            visual_obs_inputs=['*'],
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None and not backbone == "clip_feat":
            observation_space = self.obs_transform.transform_observation_space(
                observation_space, trans_keys=VISUAL_SENSORS_UUID
            )
        self._n_input = {k: 0 for k in VISUAL_SENSORS_UUID}
        visual_uuid = VISUAL_SENSORS_UUID
        if ((visual_obs_inputs is not None) and (len(visual_obs_inputs) > 0) and (visual_obs_inputs[0] != '*')):
            if isinstance(visual_obs_inputs, list):
                visual_uuid = visual_obs_inputs
            else:
                visual_uuid = visual_obs_inputs.split(MOD_SEP)

        self._id2rgb = {k: False for k in VISUAL_SENSORS_UUID}
        self._rgb_aug = None

        num_input_channels = 0
        for v_uuid in visual_uuid:
            if v_uuid in observation_space.spaces:
                if v_uuid == "semantic":
                    self._n_input[v_uuid] = 1
                elif v_uuid == "feature":
                    self._n_input[v_uuid] = observation_space.spaces[v_uuid].shape[0]

                if v_uuid == "rgb":
                    spatial_size = observation_space.spaces[v_uuid].shape[0]

                if v_uuid in ["rgb", "imagegoal_sensor_v2", "queried_image_sensor", "instance_imagegoal"]:
                    self._n_input[v_uuid] = observation_space.spaces[v_uuid].shape[2]

                if v_uuid == "depth":
                    self._n_input[v_uuid] = 1

                self._n_ouput_concat = 1
                if v_uuid == "panoramic_rgb_sensor":
                    self._n_ouput_concat = observation_space.spaces[v_uuid].shape[-1]

                num_input_channels += self._n_input[v_uuid]

        self._backbone = backbone

        restnet_type = backbone[len('fast_'):]
        backbone_enc = getattr(fast_resnet, restnet_type)

        self.encoder, v_output_shape = make_resnet_encoder(
            backbone_enc,
            num_input_channels,
            baseplanes,
            ngroups,
            spatial_size,
            normalize_visual_inputs,
            visual_encoder_embedding_size,
            film_reduction=None,
            film_layers=None,
        )

        num_compression_channels = v_output_shape[0]
        final_spatial = v_output_shape[1]

        self.output_shape = (
            num_compression_channels * self._n_ouput_concat,
            final_spatial,
            final_spatial,
        )
        self.v_output_shape = v_output_shape

    @property
    def is_blind(self):
        n_inputs = 0
        for v in self._n_input.values():
            n_inputs += v
        return n_inputs == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        cnn_input = []
        for v_uuid in self._n_input.keys():
            if self._n_input[v_uuid] > 0:
                v_observations = observations[v_uuid].float()

                if v_uuid in ["rgb", "imagegoal_sensor_v2", "panoramic_rgb_sensor", "instance_imagegoal"]:
                    v_observations = v_observations / 255.0  # normalize RGB

                # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                v_observations = v_observations.permute(0, 3, 1, 2)
                if self.obs_transform:
                    v_observations = self.obs_transform(v_observations, None)

                cnn_input.append(v_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        x = self.encoder(cnn_input)
        return x, None

class VecQc(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.num_columns = 16
        act_dim = action_space.shape[0]

        self.depth_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> [B, 32, 1]
            nn.Flatten()  # -> [B, 32]
        )

        self.action_net = nn.Sequential(
            nn.Linear(act_dim, 8),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Precompute angles
        HFOV = 90
        i = torch.arange(self.num_columns, dtype=torch.float32)
        alpha_i = torch.abs((2 * i + 1 - self.num_columns) / self.num_columns * (HFOV / 2))
        alpha_i = torch.deg2rad(alpha_i)
        self.register_buffer("alpha_i", torch.cos(alpha_i))

        self.dep_vision_range = 0.3 # 3m

    def normal_depth(self, depth_image):
        zero = torch.tensor(0, dtype=torch.float32, device=depth_image.device)
        patch = torch.tensor(1, dtype=torch.float32, device=depth_image.device)
        depth_image = torch.where(depth_image == zero, patch, depth_image)
        depth_image = torch.clamp(depth_image, max=self.dep_vision_range) / self.dep_vision_range # 3m
        depth_image = depth_image.permute(0, 3, 1, 2)
        return depth_image

    def depth2vec(self, depth_image, in_gazebo=False):
        batch_size, _, height, width = depth_image.shape
        if in_gazebo:
            crop_h = int(height * 0.4)  # 20%
            depth_image = depth_image[:, :, crop_h:-crop_h, :]

        column_width = width // self.num_columns
        min_values = torch.zeros((batch_size, self.num_columns), device=depth_image.device)

        for i in range(self.num_columns):
            col_start, col_end = i * column_width, (i + 1) * column_width
            min_values[:, i] = depth_image[:, 0, :, col_start:col_end].amin(dim=(-1, -2))

        if in_gazebo:
            return min_values
        return min_values / self.alpha_i.to(min_values.device)

    def forward(self, depth_image, action, in_gazebo=False):
        clipped_action = torch.clamp(action, min=-1, max=1)
        self.depth_image = self.normal_depth(depth_image)
        self.vec = self.depth2vec(self.depth_image, in_gazebo)

        depth_feat = self.depth_net(self.vec.unsqueeze(1))
        action_feat = self.action_net(clipped_action)

        combined = torch.cat([depth_feat, action_feat], dim=1)
        return self.classifier(combined)

class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(
        self, sample_shape = torch.Size()  # noqa: B008
    ):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x.float(), validate_args=False)


class CustomNormal(torch.distributions.normal.Normal):
    def sample(
        self, sample_shape = torch.Size()  # noqa: B008
    ):
        return self.rsample(sample_shape)

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

import math
class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        config,
    ) -> None:
        super().__init__()

        self.action_activation = config.action_activation
        self.use_softplus = config.use_softplus
        self.use_log_std = config.use_log_std
        use_std_param = config.use_std_param
        self.clamp_std = config.clamp_std

        if self.use_log_std:
            self.min_std = config.min_log_std
            self.max_std = config.max_log_std
            std_init = config.log_std_init
        elif self.use_softplus:
            inv_softplus = lambda x: math.log(math.exp(x) - 1)
            self.min_std = inv_softplus(config.min_std)
            self.max_std = inv_softplus(config.max_std)
            std_init = inv_softplus(1.0)
        else:
            self.min_std = config.min_std
            self.max_std = config.max_std
            std_init = 1.0  # initialize std value so that std ~ 1

        if use_std_param:
            self.std = torch.nn.parameter.Parameter(
                torch.randn(num_outputs) * 0.01 + std_init
            )
            num_linear_outputs = num_outputs
        else:
            self.std = None
            num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)

        if not use_std_param:
            nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], std_init)

    def forward(self, x) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        if self.std is not None:
            mu = mu_maybe_std
            std = self.std
        else:
            mu, std = torch.chunk(mu_maybe_std, 2, -1)

        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        if self.clamp_std:
            std = torch.clamp(std, self.min_std, self.max_std)
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std, validate_args=False)

class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)
