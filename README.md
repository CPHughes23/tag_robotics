<div align="center">

# 🚗 RC Car Autonomous Navigation via Reinforcement Learning

**End-to-end pipeline for training a navigation policy in simulation and deploying it on a physical RC car**

[![IsaacLab](https://img.shields.io/badge/IsaacLab-5.1-76b900?logo=nvidia&logoColor=white)](https://github.com/isaac-sim/IsaacLab)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

This project trains an RC car to navigate to target positions using Proximal Policy Optimization (PPO)
in NVIDIA Isaac Lab. The trained policy is then deployed on a physical RC car controlled via an Arduino,
with a camera-based detection system running on a laptop.

The pipeline runs across two machines:

| Machine         | Role                                                | Framework         | Acceleration               |
| --------------- | --------------------------------------------------- | ----------------- | -------------------------- |
| Desktop (Linux) | Simulation, training, evaluation                    | Isaac Lab + conda | CUDA (NVIDIA GPU)          |
| Laptop          | Camera detection, policy inference, Arduino control | venv              | MPS (Apple Silicon) or CPU |

### How It Works

```
┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│         DESKTOP                 │         │           LAPTOP                 │
│                                 │         │                                  │
│  rc_car.urdf                    │         │  Camera feed                     │
│      │                          │         │      │                           │
│      ▼                          │         │      ▼                           │
│  convert_urdf.py                │         │  Detection model                 │
│      │                          │         │      │                           │
│      ▼                          │  .pt    │      ▼                           │
│  Isaac Lab (64 parallel envs)   │ ──────► │  Policy inference                │
│      │                          │         │      │                           │
│      ▼                          │         │      ▼                           │
│  PPO training (train.py)        │         │  Arduino serial control          │
│      │                          │         │      │                           │
│      ▼                          │         │      ▼                           │
│  Checkpoint (.pt)               │         │  Physical RC car                 │
└─────────────────────────────────┘         └──────────────────────────────────┘
```

---

## Repository Structure

```
tag_robotics/
├── isaac_training/                  # Simulation and training (desktop only)
│   ├── envs/
│   │   └── rc_car_env.py            # Isaac Lab DirectRL environment
│   ├── models/
│   │   └── trained/                 # Saved checkpoints (not committed)
│   ├── requirements.txt
│   ├── train.py                     # PPO training entry point
│   ├── evaluate.py                  # Policy evaluation in Isaac Sim
│   ├── visualize_car.py             # USD model inspection tool
│   ├── convert_urdf.py              # URDF → USD converter (run once)
│   └── rc_car.urdf                  # Robot definition (source of truth)
└── robot/                           # Deployment code (laptop)
    ├── requirements.txt
    └── ...
```

---

## Environment Details

The RC car environment is implemented as a `DirectRLEnv` in Isaac Lab with:

- **64 parallel environments** running simultaneously during training
- **6 discrete actions**: forward, forward-left, forward-right, backward, backward-left, backward-right
- **6-dimensional observation space**: robot position (x, y), heading (yaw), local vector to target (x, y), distance to target
- **Shaped reward**: proximity reward + target bonus + time penalty
- **PPO** with adaptive learning rate, GAE advantage estimation, and entropy regularization

---

## Setup

### Desktop (Training)

<details>
<summary><b>Requirements</b></summary>

- Ubuntu 22.04 / 24.04
- Python 3.11
- CUDA 12.8
- Miniconda or Anaconda
- VS Code installed via snap

</details>

#### 1. Clone the repo

```bash
git clone https://github.com/CPHughes23/tag_robotics.git
cd tag_robotics
```

#### 2. Clone Isaac Lab at the pinned commit

> ⚠️ This project targets a specific Isaac Lab commit. Using a different version may cause API errors.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 87608f062bb1b3332bfbad1c44fd682abf38a088
cd ..
```

#### 3. Apply the version detection patch

Isaac Sim installed via snap reports its version differently than Isaac Lab expects.

In `IsaacLab/source/isaaclab/isaaclab/utils/version.py`, find:

```python
    version_tuple = get_version()
    return Version(f"{version_tuple[2]}.{version_tuple[3]}.{version_tuple[4]}")
```

Replace with:

```python
    version_tuple = get_version()
    major = version_tuple[2] if version_tuple[2] else "5"
    minor = version_tuple[3] if version_tuple[3] else "1"
    micro = version_tuple[4] if version_tuple[4] else "0"
    return Version(f"{major}.{minor}.{micro}")
```

#### 4. Create the conda environment

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

#### 5. Install Isaac Sim

```bash
pip install isaacsim==5.1.0 --extra-index-url https://pypi.nvidia.com
```

Accept the EULA when prompted. Then run once to complete first-time setup:

```bash
isaacsim --accept-eula
```

Wait for `Isaac Sim Full App is loaded.` then exit with `Ctrl+C`.

#### 6. Link the Isaac Sim runtime extensions

Isaac Sim installs its extensions via the VS Code snap. Link them into the conda env.

> ⚠️ The `228` in the path below is a snap revision number — run `ls ~/snap/code/` to find yours if it differs.

```bash
SNAP_EXTS="$HOME/snap/code/228/.local/share/ov/data/Kit/Isaac-Sim Full/5.1/exts/3"
CONDA_EXTS="$HOME/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts"
ln -s "$SNAP_EXTS" "$CONDA_EXTS"
```

Verify:

```bash
ls "$CONDA_EXTS" | head -5   # should show isaacsim.* folders
```

#### 7. Install PyTorch

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

#### 8. Install Isaac Lab

```bash
pip install -e IsaacLab/source/isaaclab
pip install -e IsaacLab/source/isaaclab_rl
```

#### 9. Install remaining dependencies

```bash
pip install -r isaac_training/requirements.txt
```

#### 10. Generate the USD file

The `.usd` file is not committed — generate it from the `.urdf` source:

```bash
IsaacLab/isaaclab.sh -p isaac_training/convert_urdf.py
```

#### 11. Verify the setup

```bash
# Inspect the car model (first run may take 10-20 min for shader compilation)
IsaacLab/isaaclab.sh -p isaac_training/visualize_car.py

# Quick training sanity check
IsaacLab/isaaclab.sh -p isaac_training/train.py --num_envs 4 --max_iterations 5
```

---

### Laptop (Robot Deployment)

<details>
<summary><b>Requirements</b></summary>

- Python 3.10+
- Any OS (macOS, Linux, Windows)
- No Isaac Sim needed

</details>

#### 1. Clone the repo

```bash
git clone https://github.com/CPHughes23/tag_robotics.git
cd tag_robotics
```

#### 2. Create and activate a venv

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Install PyTorch

| Hardware                 | Command                                                                |
| ------------------------ | ---------------------------------------------------------------------- |
| Apple Silicon (M1/M2/M3) | `pip install torch`                                                    |
| Intel Mac / no GPU       | `pip install torch --index-url https://download.pytorch.org/whl/cpu`   |
| NVIDIA GPU               | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |

#### 4. Install remaining dependencies

```bash
pip install -r robot/requirements.txt
```

#### 5. Verify

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('MPS:', torch.backends.mps.is_available())
"
```

---

## Usage

> All desktop scripts require `conda activate env_isaaclab` first and must be run via `isaaclab.sh`.

### Inspect the car model

```bash
IsaacLab/isaaclab.sh -p isaac_training/visualize_car.py
```

### Train a policy

```bash
IsaacLab/isaaclab.sh -p isaac_training/train.py --num_envs 64 --max_iterations 1000
```

| Argument           | Default | Description                     |
| ------------------ | ------- | ------------------------------- |
| `--num_envs`       | 64      | Number of parallel environments |
| `--max_iterations` | 1000    | Training iterations             |

Monitor with TensorBoard:

```bash
tensorboard --logdir isaac_training/models/trained/
```

### Evaluate a trained policy

```bash
IsaacLab/isaaclab.sh -p isaac_training/evaluate.py \
    --checkpoint isaac_training/models/trained/<run>/model_1000.pt \
    --num_envs 4
```

### Deploy to the laptop

```bash
# Copy checkpoint to laptop
scp isaac_training/models/trained/<run>/model_1000.pt user@laptop:~/

# Run on laptop
source .venv/bin/activate
python robot/<inference_script>.py --checkpoint ~/model_1000.pt
```

---

## Troubleshooting

**First run takes forever / looks frozen**

> Isaac Sim compiles shaders on first launch. This takes 10-20 minutes and is normal.
> Monitor GPU usage with `watch -n 2 nvidia-smi` to confirm it's working.

**`TypeError: 'NoneType' object is not callable`**

> The Isaac Sim extensions symlink is missing or broken. Redo step 6 of the desktop setup.

**`InvalidVersion: Invalid version: '..'`**

> The version detection patch (step 3) has not been applied.

**`No module named 'isaacsim'`**

> The conda environment is not activated. Run `conda activate env_isaaclab`.

**Snap revision number differs**

> Run `ls ~/snap/code/` to find your revision number and update the path in step 6.

---

## Resetting Your Environment

<details>
<summary><b>Rebuild the conda environment (desktop)</b></summary>

```bash
conda deactivate
conda env remove --name env_isaaclab
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

Then follow desktop setup from step 5. Note: the symlink in step 6 must be redone
as it lives inside the conda env folder.

</details>

<details>
<summary><b>Rebuild the venv (laptop)</b></summary>

```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

Then follow laptop setup from step 3.

</details>

---

## Dependencies

| Package   | Version    | Purpose                   |
| --------- | ---------- | ------------------------- |
| Isaac Lab | `87608f06` | Simulation framework      |
| Isaac Sim | 5.1.0      | Physics engine + renderer |
| PyTorch   | 2.7.0      | Neural network training   |
| RSL-RL    | 5.0.1      | PPO implementation        |
| OpenCV    | 4.13.0     | Camera detection          |
| PySerial  | latest     | Arduino communication     |

---

## License

This project is released under the [MIT License](LICENSE).
