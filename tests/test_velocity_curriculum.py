"""lin_vel_x_by_tracking: expands only on episode boundaries above threshold, capped."""

from types import SimpleNamespace

import torch

from cyclo_mjlab_locomotion.velocity.mdp import lin_vel_x_by_tracking


def _env(step: int, episode_sum: float):
  ranges = SimpleNamespace(lin_vel_x=(0.0, 1.0))
  return SimpleNamespace(
    common_step_counter=step,
    max_episode_length=1000,
    max_episode_length_s=20.0,
    command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))),
    reward_manager=SimpleNamespace(
      get_term_cfg=lambda _: SimpleNamespace(weight=1.2),
      _episode_sums={"track": torch.full((4,), episode_sum)},
    ),
  ), ranges


def test_lin_vel_x_by_tracking():
  ids = torch.arange(4)
  kw = dict(command_name="c", reward_name="track", max_lin_vel_x=1.5)
  env, r = _env(1000, 0.9 * 1.2 * 20.0)  # Above 0.7 x weight.
  lin_vel_x_by_tracking(env, ids, **kw)
  assert abs(r.lin_vel_x[1] - 1.1) < 1e-6
  for _ in range(10):
    lin_vel_x_by_tracking(env, ids, **kw)
  assert r.lin_vel_x == (0.0, 1.5)
  env, r = _env(1000, 0.5 * 1.2 * 20.0)  # Below threshold.
  lin_vel_x_by_tracking(env, ids, **kw)
  assert r.lin_vel_x == (0.0, 1.0)
  env, r = _env(999, 0.9 * 1.2 * 20.0)  # Not an episode boundary.
  lin_vel_x_by_tracking(env, ids, **kw)
  assert r.lin_vel_x == (0.0, 1.0)
