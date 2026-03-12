import argparse
from isaaclab.app import AppLauncher # type: ignore

parser = argparse.ArgumentParser(description="Evaluate trained RC car policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to visualize")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper # type: ignore
from envs.rc_car_env import RCCarEnv, RCCarEnvCfg

train_cfg_dict = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": 1000,
    "save_interval": 100,
    "experiment_name": "rc_car",
    "empirical_normalization": False,
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["policy"],
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 64, 32],
        "activation": "elu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 64, 32],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
}

def main():
    try:
        env_cfg = RCCarEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = RCCarEnv(cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env)

        # Load checkpoint
        checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")
        print(f"Loaded checkpoint from iteration: {checkpoint['iter']}")

        # Reconstruct the actor from saved state
        runner = OnPolicyRunner(wrapped_env, train_cfg_dict, log_dir=None, device="cuda:0")
        runner.load(args_cli.checkpoint)
        policy = runner.alg.actor
        policy.eval()

        print("Running evaluation. Watch the Isaac Sim viewport.")
        print("Press Ctrl+C to stop.")

        with torch.inference_mode():
            while simulation_app.is_running():
                obs = wrapped_env.get_observations().to("cuda:0")
                actions = runner.alg.act(obs)
                wrapped_env.step(actions)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

main()
simulation_app.close()