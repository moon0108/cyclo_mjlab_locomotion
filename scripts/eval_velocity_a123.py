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

"""Evaluate one velocity policy on A1 (flat walk), A2 (speed bins + step), A3 (turning).

Each scenario: num_envs envs, fixed seed, 12 s episodes, fixed per-env commands,
metrics over t >= 2 s and only while the env has not fallen. RMS(dv) uses body-frame xy.

  python scripts/eval_velocity_a123.py \
    --task Cyclo-Velocity-Loco-A123-K1-Rev1-v0 --checkpoint-file logs/.../model_X.pt
"""

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

EPISODE_S = 12.0
SKIP_S = 2.0
STEP_CMD_S = 2.0  # A2 step: 0 -> 1.0 m/s at this time.
SETTLE_BAND = 0.1  # +-10% of 1.0 m/s.
SMOOTH_S = 0.6  # One gait cycle; raw velocity oscillates with each step.


@dataclass
class EvalConfig:
  task: str
  checkpoint_file: str
  num_envs: int = 1024
  device: str = "cuda:0"
  seed: int = 0
  output: str | None = None
  """CSV path; defaults to eval_velocity_a123.csv next to the checkpoint."""


def main(cfg: EvalConfig) -> None:
  import mjlab.tasks  # noqa: F401
  import source.tasks  # noqa: F401
  import cyclo_mjlab_locomotion  # noqa: F401

  torch.manual_seed(cfg.seed)
  # Play cfg: no pushes, no obs noise, no curriculum, no episode timeout.
  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  cmd_cfg = env_cfg.commands["base_velocity"]
  cmd_cfg.debug_vis = False
  cmd_cfg.heading_command = False
  cmd_cfg.ranges.heading = None
  cmd_cfg.rel_standing_envs = 0.0
  cmd_cfg.rel_spin_envs = 0.0
  cmd_cfg.resampling_time_range = (1e9, 1e9)
  agent_cfg = load_rl_cfg(cfg.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=cfg.device)
  runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, strict=True,
              map_location=cfg.device)
  policy = runner.get_inference_policy(device=cfg.device)
  cmd = env.command_manager.get_term("base_velocity")
  robot = env.scene["robot"]
  n, dev = cfg.num_envs, cfg.device
  steps = round(EPISODE_S / env.step_dt)
  skip = round(SKIP_S / env.step_dt)

  def uniform(lo, hi, size=n):
    return torch.empty(size, device=dev).uniform_(lo, hi)

  def rollout(command: torch.Tensor, step_at: int | None = None):
    """command: (n, 3) [vx, vy, wz]. If step_at, vx is 0 before that step."""
    vec_env.reset()
    target = command.clone()
    cmd.vel_command_b[:] = target
    if step_at is not None:
      cmd.vel_command_b[:, 0] = 0.0
    obs = vec_env.get_observations()
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    log = {k: [] for k in ("cmd", "v", "wz", "gravity", "pos", "alive")}
    for t in range(steps):
      if t == step_at:
        cmd.vel_command_b[:, 0] = target[:, 0]
      log["cmd"].append(cmd.vel_command_b.clone())
      obs, _, _, _ = vec_env.step(policy(obs))
      fell |= env.termination_manager.terminated
      log["alive"].append(~fell)
      log["v"].append(robot.data.root_link_lin_vel_b[:, :2].clone())
      log["wz"].append(robot.data.root_link_ang_vel_b[:, 2].clone())
      log["gravity"].append(robot.data.projected_gravity_b.clone())
      log["pos"].append(robot.data.root_link_pos_w[:, :2].clone())
    return {k: torch.stack(v) for k, v in log.items()}  # (steps, n, ...)

  def rms(err: torch.Tensor, log, envs=slice(None)) -> float:
    """Pooled RMS over t >= SKIP_S, alive samples. err: (steps, n)."""
    mask = log["alive"][skip:, envs]
    return float(err[skip:, envs][mask].square().mean().sqrt())

  def vel_err(log):
    return (log["cmd"][..., :2] - log["v"]).norm(dim=-1)

  def yaw_err(log):
    return log["cmd"][..., 2] - log["wz"]

  def success(log, envs=slice(None)) -> float:
    return float(log["alive"][-1, envs].float().mean())

  rows = []  # (scenario, metric, value, target, pass)

  def add(scenario, metric, value, target=None, ok=None):
    rows.append((scenario, metric, value, target or "-", "-" if ok is None else ("PASS" if ok else "FAIL")))

  zeros = torch.zeros(n, device=dev)

  # A1: straight walk, vx ~ U(0.5, 1.0).
  log = rollout(torch.stack([uniform(0.5, 1.0), zeros, zeros], -1))
  g = log["gravity"]
  roll = torch.rad2deg(torch.atan2(-g[..., 1], -g[..., 2]))
  pitch = torch.rad2deg(torch.atan2(g[..., 0], g[..., 1:].norm(dim=-1)))
  v = rms(vel_err(log), log)
  s = success(log)
  add("A1", "RMS(dv) [m/s]", v, "<= 0.1", v <= 0.1)
  add("A1", "success rate", s, ">= 0.95", s >= 0.95)
  add("A1", "RMS(dg) roll [deg]", rms(roll, log))
  add("A1", "RMS(dg) pitch [deg]", rms(pitch, log))

  # A2: speed bins (env i -> bin i % 3).
  bins = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5))
  bin_id = torch.arange(n, device=dev) % len(bins)
  lo = torch.tensor([b[0] for b in bins], device=dev)[bin_id]
  hi = torch.tensor([b[1] for b in bins], device=dev)[bin_id]
  log = rollout(torch.stack([lo + (hi - lo) * torch.rand(n, device=dev), zeros, zeros], -1))
  err = vel_err(log)
  for i, (b_lo, b_hi) in enumerate(bins):
    envs = (bin_id == i).nonzero().squeeze(-1)
    v = rms(err, log, envs)
    add("A2", f"RMS(dv) {b_lo}-{b_hi} m/s", v, "<= 0.15", v <= 0.15)
    add("A2", f"success rate {b_lo}-{b_hi} m/s", success(log, envs))

  # A2: step 0 -> 1.0 m/s; settled once the gait-cycle mean vx stays within +-10%.
  step_at = round(STEP_CMD_S / env.step_dt)
  log = rollout(torch.stack([zeros + 1.0, zeros, zeros], -1), step_at=step_at)
  k = round(SMOOTH_S / env.step_dt)
  vx = log["v"][step_at:, :, 0].T.unsqueeze(1)  # (n, 1, T)
  vx_avg = torch.nn.functional.avg_pool1d(vx, k, stride=1).squeeze(1).T  # (T-k+1, n)
  outside = (vx_avg - 1.0).abs() > SETTLE_BAND
  # Last out-of-band index + 1, counted at the end of the averaging window.
  last_out = torch.where(outside.any(0), outside.shape[0] - outside.flip(0).int().argmax(0), 0)
  settled = (last_out < outside.shape[0]) & log["alive"][-1]
  settle_s = (last_out + k - 1) * env.step_dt
  add("A2", "step settling time [s]", float(settle_s[settled].mean()) if settled.any() else math.nan)
  add("A2", "step settled ratio", float(settled.float().mean()))

  # A3: half the envs turn in place, half walk forward while turning.
  wz = uniform(0.2, 1.0) * torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
  in_place = torch.arange(n, device=dev) < n // 2
  vx = torch.where(in_place, 0.0, uniform(0.5, 1.0))
  log = rollout(torch.stack([vx, zeros, wz], -1))
  err = yaw_err(log)
  for name, envs in (("in place", in_place), ("moving", ~in_place)):
    envs = envs.nonzero().squeeze(-1)
    w = rms(err, log, envs)
    add("A3", f"RMS(dw_yaw) {name} [rad/s]", w, "<= 0.2", w <= 0.2)
    add("A3", f"success rate {name}", success(log, envs))
  spin = in_place & log["alive"][-1]
  drift = (log["pos"][-1, spin] - log["pos"][skip, spin]).norm(dim=-1)
  add("A3", f"in-place drift {EPISODE_S - SKIP_S:.0f}s mean [m]", float(drift.mean()))
  add("A3", f"in-place drift {EPISODE_S - SKIP_S:.0f}s max [m]", float(drift.max()))

  header = ("scenario", "metric", "value", "target", "pass")
  print(f"\n{header[0]:<8} {header[1]:<32} {header[2]:>10} {header[3]:>9} {header[4]:>5}")
  for r in rows:
    print(f"{r[0]:<8} {r[1]:<32} {r[2]:>10.4f} {r[3]:>9} {r[4]:>5}")

  output = Path(cfg.output or Path(cfg.checkpoint_file).parent / "eval_velocity_a123.csv")
  with output.open("w", newline="") as f:
    csv.writer(f).writerows((header, *rows))
  print(f"[INFO] Wrote {output}")
  env.close()


if __name__ == "__main__":
  config = tyro.cli(EvalConfig)
  with torch.inference_mode():
    main(config)
