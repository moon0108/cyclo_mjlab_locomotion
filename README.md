# cyclo_mjlab_locomotion

**[Project page and interactive demo](https://moon0108.github.io/cyclo_mjlab_locomotion/)**

![K1 Rev1 walking and turning in the browser demo](https://moon0108.github.io/cyclo_mjlab_locomotion/static/images/demo.gif)

![K1 Rev1 climbing stairs in the browser demo](https://moon0108.github.io/cyclo_mjlab_locomotion/static/images/stairs.gif)

This repository provides walking and blind stair walking for the ROBOTIS AI
Sapiens K1 robot.

## Tasks

| Task ID | Description |
| --- | --- |
| `Cyclo-Velocity-Loco-A123-K1-Rev1-v0` | Flat walking: 0-1.5 m/s forward (limit grows from 1.0 m/s with tracking), +-1 rad/s yaw, 20% in-place turning |
| `Cyclo-Velocity-Loco-Stairs-K1-Rev1-v0` | 25% flat patches and 75% up/down stairs, 2-10 cm steps, flat ground at level 0. Stair envs walk at 0.3-0.8 m/s in heading mode; flat envs use the A123 commands |

## 0. Prerequisites

Install [cyclo_mjlab](https://github.com/ROBOTIS-GIT/cyclo_mjlab) first.


## 1. Installation

Install this package into the same environment as cyclo_mjlab.

```bash
pip install -e /path/to/cyclo_mjlab_locomotion
```

### Docker

The cyclo_mjlab container only mounts the cyclo_mjlab folder, at
`/workspace/cyclo_mjlab`, so a sibling folder is not visible inside it. Clone
this repository inside cyclo_mjlab on the host, and keep it out of cyclo_mjlab's
git status:

```bash
cd cyclo_mjlab
git clone https://github.com/moon0108/cyclo_mjlab_locomotion.git
echo cyclo_mjlab_locomotion/ >> .git/info/exclude
./docker/container.sh start
./docker/container.sh enter
pip install -e /workspace/cyclo_mjlab/cyclo_mjlab_locomotion
```

The install lives in the container, not in the mounted folder. Run
`pip install -e` again whenever the container is recreated, for example after
`./docker/container.sh clean`.

## Usage

The wrappers run cyclo_mjlab's own `train.py` and `play.py`, found next to the
installed `source` package, so they take the same arguments. Logs are written to
`logs/rsl_rl/` under the current folder.

### Train

```bash
python scripts/train.py Cyclo-Velocity-Loco-A123-K1-Rev1-v0 \
  --env.scene.num-envs 2048
```

`params/sim2real.yaml` is written to the run folder when training starts.

### Play

```bash
python scripts/play.py Cyclo-Velocity-Loco-Stairs-K1-Rev1-v0 \
  --checkpoint-file logs/rsl_rl/k1_velocity_stairs/<run>/model_<iteration>.pt \
  --num-envs 1
```

`exported/policy.onnx` and `exported/policy.pt` are written next to the
checkpoint. Use `--export-only True` to export without starting a viewer.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the code
derived from cyclo_mjlab and MJLab.
