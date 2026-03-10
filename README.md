# RC Car RL Training & Control

A two-machine pipeline for training an RC car navigation policy in Isaac Lab (desktop)
and deploying it on a physical RC car (MacBook).

---

## Project Structure

```
tag_robotics/
├── isaac_training/              # RL training code (desktop only)
│   ├── envs/
│   │   └── rc_car_env.py        # Isaac Lab environment definition
│   ├── models/
│   │   └── trained/             # saved checkpoints (auto-created, not committed)
│   ├── requirements.txt         # desktop dependencies
│   ├── train.py                 # PPO training script
│   ├── evaluate.py              # visualize a trained policy in Isaac Sim
│   ├── visualize_car.py         # inspect the car model in Isaac Sim
│   ├── convert_urdf.py          # convert rc_car.urdf → rc_car.usd (run once)
│   └── rc_car.urdf              # robot definition (source of truth)
└── robot/                       # detection + control code (MacBook)
    ├── requirements.txt         # MacBook dependencies
    └── ...
```

---

## Machine Roles

| Machine                 | Role                                                | Environment | GPU  |
| ----------------------- | --------------------------------------------------- | ----------- | ---- |
| Desktop (Linux)         | Isaac Sim, training, evaluation                     | conda       | CUDA |
| MacBook (Apple Silicon) | Camera detection, policy inference, Arduino control | venv        | MPS  |

---

## Desktop Setup (Training)

### Requirements

- Ubuntu 22.04 / 24.04
- Python 3.11
- CUDA 12.8
- Miniconda or Anaconda installed
- VS Code installed via snap (Isaac Sim extensions live in the snap directory)

### 1. Clone the Repo

```bash
git clone https://github.com/CPHughes23/tag_robotics.git
cd tag_robotics
```

### 2. Clone Isaac Lab at the Correct Version

This project was written against a specific Isaac Lab commit.
Using a different version may cause API errors.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 87608f062bb1b3332bfbad1c44fd682abf38a088
cd ..
```

### 3. Apply the Version Detection Patch

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

### 4. Create and Activate the Conda Environment

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

### 5. Install Isaac Sim

```bash
pip install isaacsim==5.1.0 --extra-index-url https://pypi.nvidia.com
```

Accept the EULA when prompted. Then run once to complete the first-time setup:

```bash
isaacsim --accept-eula
```

Wait for "Isaac Sim Full App is loaded." (this may take a while) then exit with Ctrl+C.

### 6. Link the Isaac Sim Extensions

Isaac Sim installs its runtime extensions via the VS Code snap. Link them into the conda env:

```bash
SNAP_EXTS="$HOME/snap/code/228/.local/share/ov/data/Kit/Isaac-Sim Full/5.1/exts/3"
CONDA_EXTS="$HOME/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts"
ln -s "$SNAP_EXTS" "$CONDA_EXTS"
```

Verify it worked:

```bash
ls "$CONDA_EXTS" | head -5
```

Should show a list of `isaacsim.*` folders.

### 7. Install PyTorch (CUDA 12.8)

Must be done before installing requirements.txt:

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 8. Install Isaac Lab

```bash
pip install -e IsaacLab/source/isaaclab
pip install -e IsaacLab/source/isaaclab_rl
```

### 9. Install Remaining Dependencies

```bash
pip install -r isaac_training/requirements.txt
```

### 10. Generate the USD File (one time only)

The USD file is not committed to the repo. Generate it from the URDF:

```bash
IsaacLab/isaaclab.sh -p isaac_training/convert_urdf.py
```

This creates `isaac_training/rc_car.usd`.
Re-run this any time you modify `rc_car.urdf`.

### 11. Verify Everything Works

Run the visualizer and let it fully load (first run takes 10-20 minutes
for shader compilation — this is normal):

```bash
IsaacLab/isaaclab.sh -p isaac_training/visualize_car.py
```

> If this is not working, try to open up a basic simulation:
>
> ```bash
> IsaacLab/isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
> ```
>
> Give it a few minutes but it should eventually let you move the camera around and you should be good after that

Then do a quick training sanity check:

```bash
IsaacLab/isaaclab.sh -p isaac_training/train.py --num_envs 4 --max_iterations 5
```

If both complete without errors, setup is complete.

---

# Laptop / Robot Machine Setup

This setup is for the machine running detection, inference, and Arduino control.
No Isaac Sim required. Works on any laptop regardless of OS.

---

## Requirements

- Python 3.10+
- No CUDA needed

---

## 1. Clone the Repo

```bash
git clone https://github.com/CPHughes23/tag_robotics.git
cd tag_robotics
```

## 2. Create and Activate a venv

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

## 3. Install PyTorch

Pick the command that matches your hardware:

**Apple Silicon (M1/M2/M3) — uses MPS GPU:**

```bash
pip install torch
```

**Intel Mac or any laptop without a GPU — CPU only:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Linux laptop with NVIDIA GPU:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Windows with NVIDIA GPU:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 4. Install Remaining Dependencies

```bash
pip install -r robot/requirements.txt
```

## 5. Verify Your Setup

Run this to confirm PyTorch is installed and check what hardware is available:

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', torch.backends.mps.is_available())
if torch.cuda.is_available():
    print('Using: CUDA')
elif torch.backends.mps.is_available():
    print('Using: MPS (Apple Silicon)')
else:
    print('Using: CPU')
"
```

---

## Notes

- CPU inference is fine for this project — the policy network is small
  ([128, 64, 32]) and runs fast even without a GPU
- If you are on Apple Silicon, MPS will be used automatically by evaluate.py
  since it detects available hardware at runtime
- The robot/requirements.txt does not include Isaac Lab or Isaac Sim —
  those are desktop-only dependencies

---

## Running the Training Scripts (Desktop Only)

Always activate the conda environment first:

```bash
conda activate env_isaaclab
```

All scripts must be launched through `isaaclab.sh`:

```bash
IsaacLab/isaaclab.sh -p <script> [args]
```

### Inspect the Car Model

Spawns the car in Isaac Sim so you can inspect joints and geometry:

```bash
IsaacLab/isaaclab.sh -p isaac_training/visualize_car.py
```

### Train a Policy

Runs PPO training. Models saved to `isaac_training/models/trained/<timestamp>/`:

```bash
IsaacLab/isaaclab.sh -p isaac_training/train.py --num_envs 64 --max_iterations 1000
```

Optional arguments:

- `--num_envs` — number of parallel environments (default: 64)
- `--max_iterations` — training iterations (default: 1000)

Monitor training with TensorBoard:

```bash
tensorboard --logdir isaac_training/models/trained/
```

### Evaluate a Trained Policy

Loads a checkpoint and runs the policy in Isaac Sim:

```bash
IsaacLab/isaaclab.sh -p isaac_training/evaluate.py \
    --checkpoint isaac_training/models/trained/<run>/<model>.pt \
    --num_envs 4
```

Optional arguments:

- `--checkpoint` — path to `.pt` checkpoint file (required)
- `--num_envs` — number of environments to visualize (default: 4)
