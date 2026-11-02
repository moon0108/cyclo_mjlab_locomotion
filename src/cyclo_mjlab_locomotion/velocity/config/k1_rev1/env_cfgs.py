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

"""Cyclo K1 Rev.1 A1-A3 configuration on top of the upstream flat task."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from source.tasks.velocity.config.k1_rev1.env_cfgs import cyclo_k1_rev1_flat_env_cfg

from cyclo_mjlab_locomotion.velocity import mdp


def cyclo_k1_rev1_flat_a123_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat config for A1-A3: forward 0-1.5 m/s, yaw +-1 rad/s, 20% in-place turns."""
  env_cfg = cyclo_k1_rev1_flat_env_cfg(play=play)

  base_cfg = env_cfg.commands["base_velocity"]
  assert isinstance(base_cfg, UniformVelocityCommandCfg)
  command_cfg = mdp.SpinVelocityCommandCfg(**vars(base_cfg), rel_spin_envs=0.2)
  # Direct yaw-rate commands instead of heading control.
  command_cfg.heading_command = False
  command_cfg.ranges.heading = None
  command_cfg.ranges.ang_vel_z = (-1.0, 1.0)
  command_cfg.ranges.lin_vel_x = (0.0, 1.5) if play else (0.0, 1.0)
  env_cfg.commands["base_velocity"] = command_cfg

  if not play:
    env_cfg.curriculum["velocity_commands"] = CurriculumTermCfg(
      func=mdp.lin_vel_x_by_tracking,
      params={
        "command_name": "base_velocity",
        "reward_name": "track_linear_velocity",
        "max_lin_vel_x": 1.5,
      },
    )

  return env_cfg
