import torch
from torch import nn as nn
from habitat import logger

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)

from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from src import resnet as fast_resnet
from src.transforms import ResizeCenterCropper
from habitat_baselines.utils.common import is_continuous_action_space
from habitat.config import Config
from torch import Size, Tensor
import torch.nn.functional as F

VISUAL_SENSORS_UUID = [
    'rgb', 'depth', 'semantic', 'feature', 'imagegoal_sensor_v2', 'panoramic_rgb_sensor', 'queried_image_sensor', 'instance_imagegoal'
]

MOD_SEP = '/'


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


def make_simplecnn_encoder(n_input_channels, spatial_size=128,
                           normalize_inputs=False, output_size=2048):
    input_shape = (spatial_size, spatial_size, n_input_channels)
    output_shape = (output_size, 1, 1)
    modules = []
    if normalize_inputs:
        modules.append(RunningMeanAndVar(n_input_channels))

    backbone = SimpleCNN(input_shape, output_size, output_shape)
    modules.append(backbone)

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
        visual_obs_inputs=['*']
    ):
        super().__init__()
        
        self.obs_transform = obs_transform
        if self.obs_transform is not None and not backbone == "clip_feat":
            logger.info(f'use obs_transform: {type(self.obs_transform)}')
            observation_space = self.obs_transform.transform_observation_space(
                observation_space, trans_keys=VISUAL_SENSORS_UUID
            )
        self._n_input = {k: 0 for k in VISUAL_SENSORS_UUID}
        logger.info('observation_space kyes: {}'
                    .format(observation_space.spaces.keys()))
        logger.info('visual_obs_inputs: {}'.format(visual_obs_inputs))
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
                logger.info("{} observation_space: {}"
                            .format(v_uuid, observation_space.spaces[v_uuid]))
                
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
        if 'simple_cnn' in backbone:
            self.encoder, v_output_shape = \
                make_simplecnn_encoder(
                    num_input_channels,
                    spatial_size,
                    normalize_visual_inputs,
                    visual_encoder_embedding_size
                )
        else:
            if 'fast_resnet' in backbone:
                restnet_type = backbone[len('fast_'):]
                backbone_enc = getattr(fast_resnet, restnet_type)
            elif 'resnet' in backbone:
                backbone_enc = getattr(resnet, backbone)
            else:
                raise ValueError('unknown type of backbone {}'
                                .format(backbone))

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
        logger.info('early fuse encoder(type: {}, in: {}, out: {})'.format(
            backbone,
            (num_input_channels, spatial_size, spatial_size),
            v_output_shape)
        )

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
        if self.is_blind:
            return None
        
        elif getattr(self, "_backbone") == "clip":
            image_goal = observations['rgb'].permute(0, 3, 1, 2) / 255.0
            # input = [self.clip_preprocessor(ig).cuda() for ig in image_goal]
            input = self.clip_preprocessor(image_goal)
            # input = torch.stack(input,dim=0)
            with torch.no_grad():
                target_encoding = self.clip_model_encoder.encode_image(input)
            x = target_encoding.type(torch.float32)
        
        elif getattr(self, "_backbone") == "clip_feat":
            x = observations['rgb'] 
            
        else:    
            seeds = None
            cnn_input = []
            for v_uuid in self._n_input.keys():
                if self._n_input[v_uuid] > 0:
                    v_observations = observations[v_uuid].float()

                    if v_uuid in ["rgb", "imagegoal_sensor_v2", "panoramic_rgb_sensor", "instance_imagegoal"]:
                        v_observations = v_observations / 255.0  # normalize RGB

                    # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                    v_observations = v_observations.permute(0, 3, 1, 2)
                    if self.obs_transform:
                        v_observations = self.obs_transform(v_observations, seeds)
                    cnn_input.append(v_observations)

            cnn_input = torch.cat(cnn_input, dim=1)
            x = self.encoder(cnn_input)
        return x, None

class VecQc(nn.Module):
    def __init__(self, action_space, safe_dis=0):
        super().__init__()
        self.num_columns = 16
        if is_continuous_action_space(action_space):
            act_dim = action_space.shape[0]
        else:
            act_dim = action_space.n

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

        self.safe_dis = safe_dis # m

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
            return min_values # [0-1] corresponds to [0-3]
        return min_values / self.alpha_i.to(min_values.device)

    def forward(self, depth_image, action, in_gazebo=False):
        clipped_action = torch.clamp(action, min=-1, max=1)
        self.depth_image = self.normal_depth(depth_image)
        self.vec = self.depth2vec(self.depth_image, in_gazebo)

        depth_feat = self.depth_net(self.vec.unsqueeze(1))
        action_feat = self.action_net(clipped_action)

        combined = torch.cat([depth_feat, action_feat], dim=1)
        return self.classifier(combined)

class CustomNormal(torch.distributions.normal.Normal):
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return self.rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> Tensor:
        return super().entropy().sum(-1, keepdim=True)

class CustomGaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        config: Config,
    ) -> None:
        super().__init__()

        self.min_std = config.min_std
        self.max_std = config.max_std

        num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)

    def forward(self, x) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        mu, std = torch.chunk(mu_maybe_std, 2, -1)

        mu = torch.tanh(mu)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std

        return CustomNormal(mu, std, validate_args=False)

def weight_init(layer: torch.nn.modules):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight.data, gain)
        if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)

def ln_activ(x: torch.Tensor, activ):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)

class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str='elu'):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)

        self.activ = getattr(F, activ)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor):
        y = ln_activ(self.l1(x), self.activ)
        y = ln_activ(self.l2(y), self.activ)
        return self.l3(y)

class MrQHead(nn.Module):
    def __init__(self, action_space, state_dim: int, action_dim: int=2,
        num_bins: int=65, zs_dim: int=128, za_dim: int=32, zsa_dim: int=128, hdim: int=128, activ: str='elu'):
        super().__init__()
        self.zs = self.mlp_zs
        self.zs_mlp = BaseMLP(state_dim, zs_dim, hdim, activ)

        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)

        self.zs_dim = zs_dim

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))


    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        dzr = self.model(zsa)
        return dzr[:,0:1], dzr[:,1:self.zs_dim+1], dzr[:,self.zs_dim+1:] # done, zs, reward

    def mlp_zs(self, state: torch.Tensor):
        return ln_activ(self.zs_mlp(state), self.activ)
