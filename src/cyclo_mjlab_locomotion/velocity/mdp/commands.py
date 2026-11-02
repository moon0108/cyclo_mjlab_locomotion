# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Velocity command with in-place turning envs and its tracking curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
  terrain_levels_vel,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "SpinVelocityCommandCfg",
  "StairsVelocityCommandCfg",
  "lin_vel_x_by_tracking",
  "stairs_terrain_levels",
)


class SpinVelocityCommand(UniformVelocityCommand):
  cfg: SpinVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    # Turn in place: zero linear command, keep the sampled yaw rate.
    spin = env_ids[torch.rand(len(env_ids), device=self.device) < self.cfg.rel_spin_envs]
    self.vel_command_b[spin, :2] = 0.0
    self.is_heading_env[spin] = False
    self.is_standing_env[spin] = False


@dataclass(kw_only=True)
class SpinVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_spin_envs: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> SpinVelocityCommand:
    return SpinVelocityCommand(self, env)


class StairsVelocityCommand(SpinVelocityCommand):
  cfg: StairsVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    # Stair columns walk forward and hold a heading; flat columns stay as sampled.
    types = self._env.scene.terrain.terrain_types[env_ids]
    stairs = env_ids[types >= self.cfg.flat_terrain_cols]
    self.vel_command_b[stairs, 0] = torch.empty(
      len(stairs), device=self.device
    ).uniform_(*self.cfg.stairs_lin_vel_x)
    self.vel_command_b[stairs, 1] = 0.0
    self.is_heading_env[stairs] = True
    self.is_standing_env[stairs] = False


@dataclass(kw_only=True)
class StairsVelocityCommandCfg(SpinVelocityCommandCfg):
  stairs_lin_vel_x: tuple[float, float]
  flat_terrain_cols: int
  """Terrain columns below this index are flat."""

  def build(self, env: ManagerBasedRlEnv) -> StairsVelocityCommand:
    return StairsVelocityCommand(self, env)


def stairs_terrain_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  flat_terrain_cols: int,
) -> dict[str, torch.Tensor]:
  """terrain_levels_vel, logging the level distribution of stair columns only."""
  terrain_levels_vel(env, env_ids, command_name)
  terrain = env.scene.terrain
  levels = terrain.terrain_levels[terrain.terrain_types >= flat_terrain_cols]
  shares = torch.bincount(levels, minlength=terrain.max_terrain_level) / len(levels)
  return {
    "mean": levels.float().mean(),
    **{f"level_{i}": share for i, share in enumerate(shares)},
  }


def lin_vel_x_by_tracking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  reward_name: str,
  max_lin_vel_x: float,
  threshold: float = 0.7,
  increment: float = 0.1,
) -> torch.Tensor:
  """Raise lin_vel_x max by `increment` once per episode length while the
  resetting envs' mean tracking reward exceeds `threshold` x its weight."""
  cfg = env.command_manager.get_term(command_name).cfg
  if len(env_ids) > 0 and env.common_step_counter % env.max_episode_length == 0:
    weight = env.reward_manager.get_term_cfg(reward_name).weight
    episode_sum = env.reward_manager._episode_sums[reward_name][env_ids]
    if episode_sum.mean() / env.max_episode_length_s > threshold * weight:
      lo, hi = cfg.ranges.lin_vel_x
      cfg.ranges.lin_vel_x = (lo, min(hi + increment, max_lin_vel_x))
  return torch.tensor(cfg.ranges.lin_vel_x[1])
