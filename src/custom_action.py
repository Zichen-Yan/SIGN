#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat_sim
from habitat.core.simulator import ActionSpaceConfiguration
from typing import Any, List, Optional, Sequence, Tuple
import attr
import numpy as np
from gym import spaces

from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.registry import registry
from habitat.core.spaces import ActionSpace

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim import RigidState
from habitat_sim.physics import VelocityControl

import magnum as mn
from magnum import Vector3

@attr.s(auto_attribs=True)
class MoveAndSpinSpec:
    forward_amount: float
    spin_amount: float

@registry.register_action_space_configuration(name="continuous")
class HabitatSimVelocityCtrlActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            "continuous_control": habitat_sim.ActionSpec(
                "continuous_control",
                MoveAndSpinSpec(forward_amount=1, spin_amount=0)
            ),
        }

@registry.register_task_action
class ContinuousAction(SimulatorTaskAction):
    name: str = "continuous_control"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        config = kwargs["config"]
        self.min_lin_vel, self.max_lin_vel = config.lin_vel_range
        self.min_ang_vel, self.max_ang_vel = config.ang_vel_range
        self.min_abs_lin_speed = config.min_abs_lin_speed
        self.min_abs_ang_speed = config.min_abs_ang_speed
        self.time_step = config.time_step

    @property
    def action_space(self):
        return ActionSpace(
            {
                "linear_velocity": spaces.Box(
                    low=np.array([-1]), # self.min_lin_vel
                    high=np.array([1]), # self.max_lin_vel
                    dtype=np.float32,
                ),
                "angular_velocity": spaces.Box(
                    low=np.array([-1]), # self.min_ang_vel
                    high=np.array([1]), # self.max_ang_vel
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        linear_velocity: float,
        angular_velocity: float,
        time_step: Optional[float] = None,
        allow_sliding: Optional[bool] = None,
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            linear_velocity: between [-1,1], scaled according to
                             config.lin_vel_range
            angular_velocity: between [-1,1], scaled according to
                             config.ang_vel_range
            time_step: amount of time to move the agent for
            allow_sliding: whether the agent will slide on collision
        """
        if allow_sliding is None:
            allow_sliding = self._sim.config.sim_cfg.allow_sliding  # type: ignore
        if time_step is None:
            time_step = self.time_step

        # Convert from [-1, 1] to [0, 1] range
        linear_velocity = (linear_velocity + 1.0) / 2.0
        angular_velocity = (angular_velocity + 1.0) / 2.0

        # Scale actions
        linear_velocity = self.min_lin_vel + linear_velocity * (
            self.max_lin_vel - self.min_lin_vel
        )
        angular_velocity = self.min_ang_vel + angular_velocity * (
            self.max_ang_vel - self.min_ang_vel
        )

        # Stop is called if both linear/angular speed are below their threshold
        if (
            abs(linear_velocity) < self.min_abs_lin_speed
            and abs(angular_velocity) < self.min_abs_ang_speed
        ):
            task.is_stop_called = True  # type: ignore
            return self._sim.get_observations_at(position=None, rotation=None)

        angular_velocity = np.deg2rad(angular_velocity)
        self.vel_control.linear_velocity = Vector3(
            [0.0, 0.0, -linear_velocity]
        )
        self.vel_control.angular_velocity = Vector3(
            [0.0, angular_velocity, 0.0]
        )
        agent_state = self._sim.get_agent_state()

        # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        current_rigid_state = RigidState(
            agent_mn_quat,
            agent_state.position,
        )

        # manually integrate the rigid state
        goal_rigid_state = self.vel_control.integrate_transform(
            time_step, current_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        if allow_sliding:
            step_fn = self._sim.pathfinder.try_step  # type: ignore
        else:
            step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore

        final_position = step_fn(
            agent_state.position, goal_rigid_state.translation
        )
        final_rotation = [
            *goal_rigid_state.rotation.vector,
            goal_rigid_state.rotation.scalar,
        ]

        # Check if a collision occured
        dist_moved_before_filter = (
            goal_rigid_state.translation - agent_state.position
        ).dot()
        dist_moved_after_filter = (final_position - agent_state.position).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the
        # filter is _less_ than the amount moved before the application of the
        # filter.
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

        agent_observations = self._sim.get_observations_at(
            position=final_position,
            rotation=final_rotation,
            keep_agent_at_new_pose=True,
        )

        # TODO: Make a better way to flag collisions
        self._sim._prev_sim_obs["collided"] = collided  # type: ignore

        return agent_observations

