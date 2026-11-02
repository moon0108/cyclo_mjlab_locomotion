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

"""Foot-relative height rewards, usable where world z is not the ground."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ("base_height_above_feet_l2", "feet_clearance_above_stance")

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height_above_feet_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize root-link height error above the lowest `asset_cfg` site."""
  asset: Entity = env.scene[asset_cfg.name]
  lowest_foot = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].min(dim=1).values
  return torch.square(asset.data.root_link_pos_w[:, 2] - lowest_foot - target_height)


def feet_clearance_above_stance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """MJLab feet_clearance with foot height measured above the lowest foot."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_z = foot_z - foot_z.min(dim=1, keepdim=True).values
  vel_norm = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  cost = torch.sum(torch.abs(foot_z - target_height) * vel_norm, dim=1)
  command = env.command_manager.get_command(command_name)
  active = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold
  return cost * active
