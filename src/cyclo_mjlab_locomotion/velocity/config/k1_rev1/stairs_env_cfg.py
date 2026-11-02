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

"""Cyclo K1 Rev.1 blind stair-walking configuration (A123 commands on stairs)."""

import math
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from source.assets.robots import K1_REV1_FOOT_SITE_NAMES

from cyclo_mjlab_locomotion.velocity import mdp

from .env_cfgs import cyclo_k1_rev1_flat_a123_env_cfg

STEP_HEIGHT_RANGE = (0.02, 0.10)  # m; rows 1-9 of the terrain curriculum.
FLAT_PROPORTION = 0.25
FLAT_ROW_DIFFICULTY = 0.1  # Row 0 of 10; sub-2 cm steps are never generated.


class _FlatFirstRow:
  """Replace the lowest curriculum row with flat ground."""

  def function(self, difficulty, spec, rng):
    if difficulty < FLAT_ROW_DIFFICULTY:
      flat = terrain_gen.BoxFlatTerrainCfg(size=self.size)
      return flat.function(difficulty, spec, rng)
    rescaled = (difficulty - FLAT_ROW_DIFFICULTY) / (1.0 - FLAT_ROW_DIFFICULTY)
    return super().function(rescaled, spec, rng)


class _PyramidStairs(_FlatFirstRow, terrain_gen.BoxPyramidStairsTerrainCfg):
  pass


class _InvertedPyramidStairs(
  _FlatFirstRow, terrain_gen.BoxInvertedPyramidStairsTerrainCfg
):
  pass


def _use_varied_play_terrain(env_cfg: ManagerBasedRlEnvCfg) -> None:
  """Respawn on a random sub-terrain at each reset, as in MJLab's G1 play."""
  terrain_generator = env_cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  terrain_generator.curriculum = False
  terrain_generator.num_rows = 5
  terrain_generator.num_cols = 5
  terrain_generator.border_width = 10.0
  env_cfg.events["randomize_terrain"] = EventTermCfg(
    func=envs_mdp.randomize_terrain,
    mode="reset",
    params={},
  )


def cyclo_k1_rev1_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat patches plus up/down stairs, trained from scratch on one policy."""
  env_cfg = cyclo_k1_rev1_flat_a123_env_cfg(play=play)

  stairs = dict(
    proportion=(1.0 - FLAT_PROPORTION) / 2,
    step_height_range=STEP_HEIGHT_RANGE,
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  env_cfg.scene.terrain.terrain_type = "generator"
  env_cfg.scene.terrain.terrain_generator = replace(
    ROUGH_TERRAINS_CFG,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=FLAT_PROPORTION),
      "pyramid_stairs": _PyramidStairs(**stairs),
      "pyramid_stairs_inv": _InvertedPyramidStairs(**stairs),
    },
    curriculum=True,
  )
  env_cfg.scene.terrain.max_init_terrain_level = 5
  flat_cols = round(FLAT_PROPORTION * ROUGH_TERRAINS_CFG.num_cols)

  # Fixed ranges: no speed curriculum.
  env_cfg.curriculum.pop("velocity_commands", None)
  env_cfg.commands["base_velocity"].ranges.lin_vel_x = (0.0, 1.5)

  # A softer fall penalty and slightly stronger yaw tracking, so that the
  # policy does not give up turning to avoid falling.
  env_cfg.rewards["termination_penalty"].weight = -50.0
  env_cfg.rewards["track_angular_velocity"].weight = 1.5

  # Heights above the lowest foot instead of world z (stair platforms sit
  # +-0.5 m off z=0), and a softer vertical-velocity penalty for climbing.
  feet = SceneEntityCfg("robot", site_names=K1_REV1_FOOT_SITE_NAMES)
  env_cfg.rewards["base_height"].func = mdp.base_height_above_feet_l2
  env_cfg.rewards["base_height"].params["asset_cfg"] = feet
  env_cfg.rewards["foot_clearance"].func = mdp.feet_clearance_above_stance
  env_cfg.rewards["lin_vel_z_l2"].weight = -0.5

  if play:
    _use_varied_play_terrain(env_cfg)
  else:
    # Stair columns get heading-hold commands; play mixes terrain types per
    # patch, so a column index means nothing there.
    command_cfg = mdp.StairsVelocityCommandCfg(
      **vars(env_cfg.commands["base_velocity"]),
      stairs_lin_vel_x=(0.3, 0.8),
      flat_terrain_cols=flat_cols,
    )
    command_cfg.heading_command = True
    command_cfg.ranges.heading = (-math.pi, math.pi)
    command_cfg.rel_heading_envs = 0.0
    env_cfg.commands["base_velocity"] = command_cfg
    env_cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.stairs_terrain_levels,
      params={"command_name": "base_velocity", "flat_terrain_cols": flat_cols},
    )

  return env_cfg
