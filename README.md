# cyclo_mjlab_locomotion

**[프로젝트 페이지 및 인터랙티브 데모](https://moon0108.github.io/cyclo_mjlab_locomotion/)**

![브라우저 데모에서 걷고 회전하는 K1 Rev1](https://moon0108.github.io/cyclo_mjlab_locomotion/static/images/demo.gif)

![브라우저 데모에서 계단을 오르는 K1 Rev1](https://moon0108.github.io/cyclo_mjlab_locomotion/static/images/stairs.gif)

이 레포는 ROBOTIS AI Sapiens K1 보행과 블라인드 계단 보행을 위한 코드베이스입니다.

## 태스크

| 태스크 ID | 설명 |
| --- | --- |
| `Cyclo-Velocity-Loco-A123-K1-Rev1-v0` | 평지 보행 학습 |
| `Cyclo-Velocity-Loco-Stairs-K1-Rev1-v0` | 계단 보행 학습. 계단 높이 2-10 cm |

## 0. 사전 설치

먼저 [cyclo_mjlab](https://github.com/ROBOTIS-GIT/cyclo_mjlab)을 설치하세요.


## 1. 설치

cyclo_mjlab과 같은 환경에 이 패키지를 설치하세요.

```bash
pip install -e /path/to/cyclo_mjlab_locomotion
```

### Docker

호스트에서 이 레포를 cyclo_mjlab 폴더 안에
clone하고, cyclo_mjlab의 git status에 나타나지 않도록 제외하세요.

```bash
cd cyclo_mjlab
git clone https://github.com/moon0108/cyclo_mjlab_locomotion.git
echo cyclo_mjlab_locomotion/ >> .git/info/exclude
./docker/container.sh start
./docker/container.sh enter
pip install -e /workspace/cyclo_mjlab/cyclo_mjlab_locomotion
```

설치 정보는 마운트된 폴더가 아니라 컨테이너 안에 저장됩니다. `./docker/container.sh clean`
등으로 컨테이너를 다시 만들 때마다 `pip install -e`를 다시 실행하세요.

## 사용법

이 레포의 스크립트는 설치된 `source` 패키지 위치에서 cyclo_mjlab의 `train.py`와 `play.py`를
찾아 그대로 실행하므로, 인자도 같습니다. 로그는 현재 폴더의 `logs/rsl_rl/`에 저장됩니다.

### 학습

```bash
python scripts/train.py Cyclo-Velocity-Loco-A123-K1-Rev1-v0 \
  --env.scene.num-envs 2048
```

학습을 시작하면 실행 폴더에 `params/sim2real.yaml`이 생성됩니다.

### 실행

```bash
python scripts/play.py Cyclo-Velocity-Loco-Stairs-K1-Rev1-v0 \
  --checkpoint-file logs/rsl_rl/k1_velocity_stairs/<run>/model_<iteration>.pt \
  --num-envs 1
```

체크포인트와 같은 폴더에 `exported/policy.onnx`와 `exported/policy.pt`가 생성됩니다.
뷰어 없이 내보내기만 하려면 `--export-only True`를 사용하세요.

## 라이선스

Apache License 2.0을 따릅니다. cyclo_mjlab과 MJLab에서 가져온 코드는 [LICENSE](LICENSE)와
[NOTICE](NOTICE)를 참고하세요.
