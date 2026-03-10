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
git clone <your-github-url>
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

Wait for "Isaac Sim Full App is loaded." then exit with Ctrl+C.

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

Then do a quick training sanity check:

```bash
IsaacLab/isaaclab.sh -p isaac_training/train.py --num_envs 4 --max_iterations 5
```

If both complete without errors, setup is complete.

---

## MacBook Setup (Robot)

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- No Isaac Sim needed

### 1. Clone the Repo

```bash
git clone <your-github-url>
cd tag_robotics
```

### 2. Create and Activate the venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch (Apple Silicon MPS)

MPS support is built into the standard PyTorch package on Apple Silicon:

```bash
pip install torch
```

### 4. Install Remaining Dependencies

```bash
pip install -r robot/requirements.txt
```

### 5. Verify MPS is Available

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Should print `True`. If it prints `False`, make sure you are on macOS 12.3+.

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
    --checkpoint isaac_training/models/trained/<run>/model_1000.pt \
    --num_envs 4
```

Optional arguments:

- `--checkpoint` — path to `.pt` checkpoint file (required)
- `--num_envs` — number of environments to visualize (default: 4)

---

## Deploying to the MacBook

Copy a trained checkpoint from the desktop to the MacBook:

```bash
scp isaac_training/models/trained/<run>/model_1000.pt user@macbook:~/
```

Then run inference on the MacBook:

```bash
source .venv/bin/activate
python robot/<your_inference_script>.py --checkpoint ~/model_1000.pt
```

PyTorch automatically uses MPS on Apple Silicon. No Isaac Sim required.

---

## Regenerating the Conda Environment from Scratch

```bash
conda deactivate
conda env remove --name env_isaaclab
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

Then follow the desktop setup steps from step 5 onwards.
Note: you will need to redo the symlink in step 6 as it lives inside the conda env.

## Regenerating the MacBook venv from Scratch

```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

Then follow the MacBook setup steps from step 3 onwards.

---

## Notes

- `rc_car.usd` is not committed — generate it with `convert_urdf.py`
- `models/trained/` is not committed — checkpoints can be large
- The exts symlink lives inside the conda env and must be recreated
  if the conda env is deleted and rebuilt
- First run of any Isaac Sim script takes 10-20 minutes for shader
  compilation — subsequent runs are fast
- Desktop uses conda because Isaac Sim's Omniverse runtime requires it
- MacBook uses a plain venv since it only needs lightweight inference
  dependencies
- Do not try to share one environment between machines
