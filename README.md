# RC Car RL Training & Control

A two-machine pipeline for training an RC car navigation policy in Isaac Lab (desktop)
and deploying it on a physical RC car (MacBook).

---

## Project Structure

```
tag_robotics/
├── isaac_training/         # RL training code (desktop only)
│   ├── envs/
│   │   └── rc_car_env.py   # Isaac Lab environment definition
│   ├── models/
│   │   └── trained/        # saved checkpoints (auto-created, not committed)
│   ├── requirements.txt    # desktop dependencies
│   ├── train.py            # PPO training script
│   ├── evaluate.py         # visualize a trained policy in Isaac Sim
│   ├── visualize.py        # inspect the car model in Isaac Sim
│   ├── convert_urdf.py     # convert rc_car.urdf → rc_car.usd (run once)
│   └── rc_car.urdf         # robot definition (source of truth)
└── robot/                  # detection + control code (MacBook)
    ├── requirements.txt    # MacBook dependencies
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

- Ubuntu 22.04
- Python 3.11
- CUDA 12.8
- Miniconda or Anaconda installed
- Isaac Sim 5.1.0 installed via conda

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

### 3. Set Up the Conda Environment

Create a fresh conda environment with Python 3.11:

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

### 4. Install Isaac Sim

Isaac Sim must be installed via NVIDIA's pip index into the conda environment:

```bash
pip install isaacsim==5.1.0.0 --extra-index-url https://pypi.nvidia.com
```

Accept the EULA when prompted.

### 5. Install PyTorch (CUDA 12.8)

Must be done before installing requirements.txt:

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 6. Install Isaac Lab

```bash
pip install -e path/to/IsaacLab/source/isaaclab
pip install -e path/to/IsaacLab/source/isaaclab_rl
```

### 7. Install Remaining Dependencies

```bash
pip install -r isaac_training/requirements.txt
```

### 8. Generate the USD File (one time only)

The USD file is not committed to the repo. Generate it from the URDF:

```bash
python isaac_training/convert_urdf.py
```

This creates `isaac_training/rc_car.usd`.
Re-run this any time you modify `rc_car.urdf`.

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

All scripts must be launched through `isaaclab.sh` so Isaac Sim loads correctly:

```bash
path/to/IsaacLab/isaaclab.sh -p <script>
```

### Inspect the Car Model

Spawns the car in Isaac Sim so you can inspect joints and geometry.
No training, no actions — just a visual check.

```bash
../IsaacLab/isaaclab.sh -p isaac_training/visualize.py
```

### Train a Policy

Runs PPO training with 64 parallel environments.
Models are saved to `isaac_training/models/trained/<timestamp>/`.

```bash
../IsaacLab/isaaclab.sh -p isaac_training/train.py --num_envs 64 --max_iterations 1000
```

Optional arguments:

- `--num_envs` — number of parallel environments (default: 64)
- `--max_iterations` — training iterations (default: 1000)

Monitor training with TensorBoard:

```bash
tensorboard --logdir isaac_training/models/trained/
```

### Evaluate a Trained Policy

Loads a checkpoint and runs the policy in Isaac Sim so you can watch it:

```bash
../IsaacLab/isaaclab.sh -p isaac_training/evaluate.py \
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

Then run the robot code on the MacBook using the venv:

```bash
source .venv/bin/activate
python robot/<your_inference_script>.py --checkpoint ~/model_1000.pt
```

The MacBook runs only the actor network for inference — no Isaac Sim required.
PyTorch will automatically use MPS for GPU-accelerated inference on Apple Silicon.

---

## Regenerating the Conda Environment from Scratch

If the desktop environment gets into a bad state:

```bash
conda deactivate
conda env remove --name env_isaaclab
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

Then follow the desktop setup steps from step 4 onwards.

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
- All hard-coded paths use `__file__`-relative paths so no path editing
  is needed after cloning
- Desktop uses conda because Isaac Sim's full Omniverse runtime requires it
- MacBook uses a plain venv since it only needs lightweight inference dependencies
- The two environments are intentionally separate — do not try to share one
  environment between machines
