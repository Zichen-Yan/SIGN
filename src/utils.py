from pathlib import Path
import logging
import os
import shutil
import yaml
from datetime import datetime
from glob import glob

import habitat
import torch
from habitat_baselines.config.default import get_config as get_habitat_config
import torch.nn.functional as F

logger = logging.getLogger(__name__)


from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoalSensor
)
from src.sensor import (
    ImageGoalSensorV2,
    QueriedImageSensor
)
from habitat.config import Config

def compare_config(cc, pc):
    def flat_map(src, target=None, prefix=""):
        if target is None:
            target = {}
        for k, v in src.items():
            if type(v) is dict:
                flat_map(v, target, prefix + k + ".")
            else:
                target[prefix+k] = v
        return target

    # load previous config and find unmatch config
    cc = yaml.load(str(cc), Loader=yaml.FullLoader)
    pc = yaml.load(str(pc), Loader=yaml.FullLoader)
    unmatch_config = {k:v for k,v in flat_map(pc).items() if flat_map(cc)[k] != v}
    return unmatch_config

def save_config(run_dir, config, type):
    if config is not None:
        F = open(run_dir / 'config_of_{}.yaml'.format(type), 'w')
        F.write(str(config))
        F.close()

def get_random_rundir(exp_dir: Path, prefix: str = 'run', suffix: str = ''):
    if exp_dir.exists():
        runs = glob(str(exp_dir / '*_run*'))
        num_runs = len([r for r in runs if (prefix + '_') in r])
    else:
        num_runs = 0
    dt = datetime.now().strftime('%Y%m%d-%H%M%S')
    rundir = prefix + '_' + 'run{}'.format(num_runs) + '_' + suffix
    return (exp_dir / rundir)

def change_habitat_config(config):
    with habitat.config.read_write(config):
        if ImageGoalSensorV2.cls_uuid in config.habitat.task.sensors:
            goalsensoruuid = ImageGoalSensorV2.cls_uuid
        elif QueriedImageSensor.cls_uuid in config.habitat.task.sensors:
            goalsensoruuid = QueriedImageSensor.cls_uuid  
        elif InstanceImageGoalSensor.cls_uuid + "_sensor" in config.habitat.task.sensors:
            goalsensoruuid = InstanceImageGoalSensor.cls_uuid
        else:
            assert False,"Do not specifit goal sensor"

        config.habitat.task.distance_to_view = Config()
        config.habitat.task.distance_to_view.type = "DistanceToView"
        config.habitat.task.distance_to_view.goalsensoruuid = goalsensoruuid

        config.habitat.task.view_match = Config()
        config.habitat.task.view_match.type = "ViewMatch"
        config.habitat.task.view_match.view_weight = 0.5
        config.habitat.task.view_match.angle_threshold = 25.0
        config.habitat.task.view_match.goalsensoruuid = goalsensoruuid

        config.habitat.task.view_angle = Config()
        config.habitat.task.view_angle.type = "ViewAngle"
        config.habitat.task.view_angle.goalsensoruuid = goalsensoruuid
    return config


def get_config(exp_config, opts, run_type, model_dir, overwrite, note, debug):
    exp_config = exp_config.split(',')
    exp_config = [path if '.yaml' in path else f'src/config/{path}.yaml' for path in exp_config]

    config = get_habitat_config(exp_config, opts)
    config = change_habitat_config(config)
    config.defrost()

    if model_dir == None:
        model_dir = 'results/official'

    config.habitat_baselines.checkpoint_folder = os.path.join(model_dir, 'ckpts')
    if 'habitat_baselines.eval_ckpt_path_dir' not in opts:
        config.habitat_baselines.eval_ckpt_path_dir = os.path.join(model_dir, 'ckpts')
    config.habitat_baselines.tensorboard_dir = os.path.join(model_dir, 'tb')
    config.habitat_baselines.video_dir = os.path.join(model_dir, "video")

    if config.habitat.simulator.action_space_config == "continuous":
        config.habitat.task.possible_actions = ["continuous_control"]

    if overwrite:
        logger.warning('Warning! overwrite is specified!\nCurrent model dir will be removed!')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
            os.makedirs(model_dir, exist_ok=True)

    run_dir = get_random_rundir(exp_dir=Path(model_dir), prefix=run_type, suffix=note)

    config.habitat_baselines.log_file = os.path.join(run_dir, 'log.txt')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config.habitat_baselines.tensorboard_dir, exist_ok=True)
    os.makedirs(config.habitat_baselines.checkpoint_folder, exist_ok=True)

    save_config(run_dir, config, run_type)

    config.freeze()
    return config

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class TwoHot:
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp
        self.num_bins = num_bins


    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot


    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)


    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)