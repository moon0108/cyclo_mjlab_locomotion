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

from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from source.tasks.velocity.config.k1_rev1.rl_cfg import robotis_k1_ppo_runner_cfg
from source.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import cyclo_k1_rev1_flat_a123_env_cfg
from .stairs_env_cfg import cyclo_k1_rev1_stairs_env_cfg


register_mjlab_task(
  task_id="Cyclo-Velocity-Loco-A123-K1-Rev1-v0",
  env_cfg=cyclo_k1_rev1_flat_a123_env_cfg(),
  play_env_cfg=cyclo_k1_rev1_flat_a123_env_cfg(play=True),
  rl_cfg=replace(robotis_k1_ppo_runner_cfg(), experiment_name="k1_velocity_a123"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Cyclo-Velocity-Loco-Stairs-K1-Rev1-v0",
  env_cfg=cyclo_k1_rev1_stairs_env_cfg(),
  play_env_cfg=cyclo_k1_rev1_stairs_env_cfg(play=True),
  rl_cfg=replace(robotis_k1_ppo_runner_cfg(), experiment_name="k1_velocity_stairs"),
  runner_cls=VelocityOnPolicyRunner,
)
